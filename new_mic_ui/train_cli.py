#!/usr/bin/env python3
"""
train_cli.py — local entry point for the AWS training pipeline.

Commands:
  upload    Sync a local dataset folder to S3
  launch    Package code, launch EC2 spot instance, stream training logs
  status    Stream logs for a running/finished job
  download  Download a trained model adapter from S3
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
import yaml

HERE = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── shared helpers ────────────────────────────────────────────────────────────

def get_region(explicit: str = None) -> str:
    if explicit:
        return explicit
    session = boto3.session.Session()
    return session.region_name or "us-east-1"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── upload ────────────────────────────────────────────────────────────────────

def cmd_upload(args):
    cfg = load_config(args.config)
    bucket = args.bucket or cfg["data"]["bucket"]
    prefix = cfg["data"]["dataset_prefix"].rstrip("/")
    dataset_root = Path(args.dataset)

    if not dataset_root.exists():
        sys.exit(f"Dataset path not found: {dataset_root}")

    s3 = boto3.client("s3", region_name=get_region(args.region))

    # Collect all class folders (direct subdirectories)
    classes = sorted(d.name for d in dataset_root.iterdir() if d.is_dir())
    print(f"Found classes: {classes}")

    total = 0
    for cls in classes:
        cls_dir = dataset_root / cls
        images = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        print(f"\n{cls}: {len(images)} images → s3://{bucket}/{prefix}/{cls}/")
        for i, img in enumerate(images, 1):
            key = f"{prefix}/{cls}/{img.name}"
            s3.upload_file(str(img), bucket, key)
            if i % 500 == 0:
                print(f"  {i}/{len(images)}")
        total += len(images)

    print(f"\nUploaded {total} images to s3://{bucket}/{prefix}/")


# ── launch ────────────────────────────────────────────────────────────────────

def _find_deep_learning_ami(ec2, region: str) -> tuple:
    """Return (ami_id, ami_name) for the latest PyTorch Deep Learning AMI."""
    for pattern in [
        "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04)*",
        "Deep Learning AMI GPU PyTorch 2.* (Ubuntu 22.04)*",
        "Deep Learning AMI (Ubuntu 22.04) Version*",
    ]:
        resp = ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": [pattern]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        imgs = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
        if imgs:
            return imgs[0]["ImageId"], imgs[0]["Name"]
    sys.exit("Could not find a Deep Learning AMI in this region. Set --ami manually.")


def _ensure_iam_role(iam, bucket: str) -> str:
    role_name = "MIC2TrainingRole"
    profile_name = "MIC2TrainingProfile"

    # Role trust policy
    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    # Inline policy: S3 access to training bucket only
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"],
            },
        ],
    })

    # Create role if it doesn't exist
    try:
        iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
        print(f"Created IAM role: {role_name}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using existing IAM role: {role_name}")

    # Put inline policy
    iam.put_role_policy(RoleName=role_name, PolicyName="S3Access", PolicyDocument=policy)

    # Create instance profile
    try:
        iam.create_instance_profile(InstanceProfileName=profile_name)
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
        print(f"Created instance profile: {profile_name}")
        time.sleep(10)  # IAM propagation delay
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using existing instance profile: {profile_name}")

    return profile_name


def _pick_vpc_and_subnets(ec2, instance_type: str) -> tuple:
    """
    Return (vpc_id, [subnet_ids]) choosing the VPC that has the most subnets
    in AZs that support the requested instance type.
    """
    az_resp = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    )
    supported_azs = {o["Location"] for o in az_resp.get("InstanceTypeOfferings", [])}

    all_subnets = ec2.describe_subnets()["Subnets"]

    # Group subnets by VPC, keeping only those in supported AZs
    from collections import defaultdict
    vpc_subnets = defaultdict(list)
    for s in all_subnets:
        if s["AvailabilityZone"] in supported_azs:
            vpc_subnets[s["VpcId"]].append(s)

    if not vpc_subnets:
        # Fall back: any VPC, any subnet
        for s in all_subnets:
            vpc_subnets[s["VpcId"]].append(s)

    if not vpc_subnets:
        sys.exit("No subnets found. Pass --subnet explicitly.")

    # Pick VPC with most qualifying subnets (prefer default VPC on tie)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "state", "Values": ["available"]}])["Vpcs"]
    default_vpc = next((v["VpcId"] for v in vpcs if v.get("IsDefault")), None)

    best_vpc = max(
        vpc_subnets,
        key=lambda vid: (len(vpc_subnets[vid]), vid == default_vpc),
    )
    subnet_ids = [s["SubnetId"] for s in
                  sorted(vpc_subnets[best_vpc], key=lambda s: s["AvailabilityZone"])]
    print(f"VPC: {best_vpc} — {len(subnet_ids)} subnet(s) in supported AZs: "
          f"{[s['AvailabilityZone'] for s in sorted(vpc_subnets[best_vpc], key=lambda s: s['AvailabilityZone'])]}")
    return best_vpc, subnet_ids


def _get_or_create_sg(ec2, vpc_id: str) -> str:
    sg_name = "mic2-training-sg"
    # Look for existing SG in the target VPC
    resp = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [sg_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )
    sgs = resp.get("SecurityGroups", [])
    if sgs:
        print(f"Using existing security group: {sgs[0]['GroupId']}")
        return sgs[0]["GroupId"]

    sg = ec2.create_security_group(
        GroupName=sg_name,
        Description="MIC2 training outbound only",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    print(f"Created security group: {sg_id} (VPC: {vpc_id})")
    return sg_id


def _package_training_code() -> bytes:
    """Tar gz the EC2-side training files."""
    buf = io.BytesIO()
    ec2_files = ["model.py", "dataset.py", "train.py", "requirements_ec2.txt"]
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in ec2_files:
            fpath = HERE / fname
            if not fpath.exists():
                sys.exit(f"Missing training file: {fpath}")
            tar.add(str(fpath), arcname=fname)
    return buf.getvalue()


def _build_bootstrap(bucket: str, job_name: str, region: str) -> str:
    return f"""#!/bin/bash
exec > /var/log/training.log 2>&1

BUCKET="{bucket}"
JOB_NAME="{job_name}"
REGION="{region}"
TRAIN_PID=""

# ── Exit handler: upload final log + status, then shut down ──────────────────
cleanup() {{
    local EXIT_CODE=${{1:-$?}}
    # Read status file if training wrote one, otherwise derive from exit code
    if [ -f /tmp/job_status.txt ]; then
        STATUS=$(cat /tmp/job_status.txt)
    elif [ $EXIT_CODE -eq 0 ]; then
        STATUS="SUCCESS"
    else
        STATUS="FAILED:$EXIT_CODE"
    fi
    echo "=== Cleanup: $STATUS at $(date) ==="
    # Final log upload (retry once to ensure the complete log is in S3)
    aws s3 cp /var/log/training.log s3://$BUCKET/jobs/$JOB_NAME/training.log --region $REGION 2>/dev/null || true
    sleep 5
    aws s3 cp /var/log/training.log s3://$BUCKET/jobs/$JOB_NAME/training.log --region $REGION 2>/dev/null || true
    echo "$STATUS" | aws s3 cp - s3://$BUCKET/jobs/$JOB_NAME/status.txt --region $REGION 2>/dev/null || true
    if echo "$STATUS" | grep -q "^FAILED"; then
        # Keep the instance alive for 30 minutes on failure so you can SSH in and inspect
        echo "=== FAILED — keeping instance alive for 30 min for debugging ==="
        sleep 1800
    fi
    echo "=== Shutting down at $(date) ==="
    shutdown -h now
}}
trap 'cleanup $?' EXIT

# ── Spot-interruption watcher ─────────────────────────────────────────────────
# AWS gives 2-minute notice via the metadata endpoint.
# Send SIGTERM to the training process so it saves a checkpoint, then wait.
spot_watcher() {{
    while true; do
        TERM_TIME=$(curl -sf --max-time 2 \
            http://169.254.169.254/latest/meta-data/spot/termination-time 2>/dev/null)
        if [ -n "$TERM_TIME" ]; then
            echo "=== SPOT INTERRUPTION NOTICE: $TERM_TIME ==="
            echo "INTERRUPTED" > /tmp/job_status.txt
            aws s3 cp /var/log/training.log s3://$BUCKET/jobs/$JOB_NAME/training.log \
                --region $REGION --quiet 2>/dev/null || true
            echo "INTERRUPTED" | aws s3 cp - s3://$BUCKET/jobs/$JOB_NAME/status.txt \
                --region $REGION 2>/dev/null || true
            if [ -n "$TRAIN_PID" ]; then
                echo "Sending SIGTERM to training process $TRAIN_PID …"
                kill -SIGTERM "$TRAIN_PID" 2>/dev/null
                # Wait up to 90s for the checkpoint to be saved
                for i in $(seq 1 18); do
                    sleep 5
                    kill -0 "$TRAIN_PID" 2>/dev/null || break
                done
            fi
            break
        fi
        sleep 5
    done
}}

echo "=== Instance started: $(date) ==="
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

# ── Find Python with PyTorch ──────────────────────────────────────────────────
echo "=== Diagnosing Python/conda environment ==="
echo "PATH: $PATH"
find /opt /home/ubuntu -maxdepth 6 -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -3 | xargs -I{{}} echo "  conda.sh: {{}}"
find /opt /home/ubuntu -maxdepth 4 -name "python3" -type f 2>/dev/null | head -10 | xargs -I{{}} echo "  python3: {{}}"

# Source whichever conda init script exists
for CONDA_SH in \
    /opt/conda/etc/profile.d/conda.sh \
    /home/ubuntu/anaconda3/etc/profile.d/conda.sh \
    /home/ubuntu/miniconda3/etc/profile.d/conda.sh \
    $(find /opt /home/ubuntu -maxdepth 6 -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null | head -1); do
    if [ -f "$CONDA_SH" ]; then
        echo "Sourcing: $CONDA_SH"
        source "$CONDA_SH"
        break
    fi
done

# Try activating common DL AMI env names
for ENV in pytorch pytorch_p311 pytorch_p310 base; do
    conda activate "$ENV" 2>/dev/null && echo "Activated conda env: $ENV" && break || true
done

# Find first Python that has torch
PYTHON=""
for P in \
    $(which python3 2>/dev/null) \
    $(which python 2>/dev/null) \
    /opt/conda/bin/python3 \
    $(find /opt /home/ubuntu -maxdepth 6 -name python3 -type f 2>/dev/null); do
    if [ -n "$P" ] && "$P" -c "import torch" 2>/dev/null; then
        PYTHON="$P"
        echo "Found Python with PyTorch: $PYTHON"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "PyTorch not found in any Python. Installing via pip into system Python..."
    PIP_PYTHON=$(which python3 || which python)
    $PIP_PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q
    PYTHON="$PIP_PYTHON"
fi

echo "Python: $PYTHON ($($PYTHON --version))"
echo "PyTorch CUDA: $($PYTHON -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

# Background: upload log to S3 every 30s for live streaming
(while true; do
    aws s3 cp /var/log/training.log s3://$BUCKET/jobs/$JOB_NAME/training.log \
        --region $REGION --quiet 2>/dev/null || true
    sleep 30
done) &

# Start spot watcher in background
spot_watcher &

# Download + extract training code
echo "=== Downloading code ==="
aws s3 cp s3://$BUCKET/jobs/$JOB_NAME/code.tar.gz /tmp/code.tar.gz --region $REGION
mkdir -p /opt/training
tar -xzf /tmp/code.tar.gz -C /opt/training/

# Install extra deps
echo "=== Installing dependencies ==="
$PYTHON -m pip install -q -r /opt/training/requirements_ec2.txt

# Download config
aws s3 cp s3://$BUCKET/jobs/$JOB_NAME/config.yaml /opt/training/config.yaml --region $REGION

# Train — run in foreground so we capture its PID for the spot watcher
echo "=== Starting training: $(date) ==="
cd /opt/training
PYTHONUNBUFFERED=1 $PYTHON -u train.py --config config.yaml --bucket $BUCKET &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
wait $TRAIN_PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "SUCCESS" > /tmp/job_status.txt
    echo "=== Training complete: $(date) ==="
else
    echo "FAILED:$TRAIN_EXIT" > /tmp/job_status.txt
    echo "=== Training failed (exit $TRAIN_EXIT): $(date) ===" >&2
fi
"""


def cmd_launch(args):
    cfg = load_config(args.config)
    bucket = args.bucket or cfg["data"]["bucket"]
    if not bucket:
        sys.exit("Bucket required: set data.bucket in config or pass --bucket")

    job_name = cfg["job_name"]
    region = get_region(args.region)
    instance_type = args.instance_type or cfg.get("instance", {}).get("type", "g5.xlarge")
    max_price = str(cfg.get("instance", {}).get("max_spot_price", "1.50"))

    ec2 = boto3.client("ec2", region_name=region)
    iam = boto3.client("iam", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    # ── Clear stale status/log from any previous run ─────────────────────────
    # (keeps checkpoints intact so training can resume from them)
    for stale_key in [f"jobs/{job_name}/status.txt", f"jobs/{job_name}/training.log"]:
        try:
            boto3.client("s3", region_name=region).delete_object(Bucket=bucket, Key=stale_key)
        except Exception:
            pass

    # ── Upload code + config ─────────────────────────────────────────────────
    print("=== Packaging training code ===")
    code_bytes = _package_training_code()
    code_key = f"jobs/{job_name}/code.tar.gz"
    cfg_key = f"jobs/{job_name}/config.yaml"

    print(f"Uploading code ({len(code_bytes)//1024} KB) → s3://{bucket}/{code_key}")
    s3.put_object(Bucket=bucket, Key=code_key, Body=code_bytes)

    # Patch bucket into config if missing and re-upload
    cfg["data"]["bucket"] = bucket
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name
    s3.upload_file(tmp_path, bucket, cfg_key)
    os.unlink(tmp_path)
    print(f"Config uploaded → s3://{bucket}/{cfg_key}")

    # ── IAM ──────────────────────────────────────────────────────────────────
    print("=== Setting up IAM ===")
    try:
        profile_name = _ensure_iam_role(iam, bucket)
    except Exception as e:
        sys.exit(
            f"IAM setup failed: {e}\n"
            "Create role 'MIC2TrainingRole' manually with S3 access and an instance profile."
        )

    # ── AMI ──────────────────────────────────────────────────────────────────
    ami_id = args.ami
    if not ami_id:
        print("=== Finding Deep Learning AMI ===")
        ami_id, ami_name = _find_deep_learning_ami(ec2, region)
        print(f"AMI: {ami_id} ({ami_name})")

    # ── Security group + subnet ───────────────────────────────────────────────
    vpc_id, auto_subnets = _pick_vpc_and_subnets(ec2, instance_type)
    sg_id = _get_or_create_sg(ec2, vpc_id)

    # ── Launch spot instance — try all available AZs until one accepts ────────
    bootstrap = _build_bootstrap(bucket, job_name, region)
    user_data = base64.b64encode(bootstrap.encode()).decode()

    on_demand = getattr(args, "on_demand", False)
    subnets = [args.subnet] if args.subnet else auto_subnets
    mode = "on-demand" if on_demand else "spot"
    print(f"=== Launching {instance_type} {mode} instance (trying {len(subnets)} AZ(s)) ===")

    instance_id = None
    last_error = None
    for subnet_id in subnets:
        try:
            launch_kwargs = dict(
                ImageId=ami_id,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet_id,
                SecurityGroupIds=[sg_id],
                IamInstanceProfile={"Name": profile_name},
                UserData=user_data,
                BlockDeviceMappings=[{
                    "DeviceName": "/dev/sda1",
                    "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True},
                }],
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"training-{job_name}"},
                        {"Key": "MIC2Job", "Value": job_name},
                    ],
                }],
            )
            if not on_demand:
                launch_kwargs["InstanceMarketOptions"] = {
                    "MarketType": "spot",
                    "SpotOptions": {
                        "MaxPrice": max_price,
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                }
            resp = ec2.run_instances(**launch_kwargs)
            instance_id = resp["Instances"][0]["InstanceId"]
            print(f"Launched in subnet {subnet_id}")
            break
        except Exception as e:
            print(f"  {subnet_id}: {e}")
            last_error = e

    if not instance_id:
        sys.exit(f"Could not launch in any AZ. Last error: {last_error}")
    print(f"\nInstance launched: {instance_id}")
    print(f"Region:   {region}")
    print(f"Job:      {job_name}")
    print(f"Logs:     s3://{bucket}/jobs/{job_name}/training.log")
    print(f"Model:    s3://{bucket}/{cfg['output']['prefix']}/{job_name}/")
    print(f"\nStream logs: python train_cli.py status --config {args.config} --bucket {bucket}")

    if not args.no_wait:
        _stream_logs(s3, bucket, job_name, instance_id, ec2, region,
                     config_path=args.config,
                     output_prefix=cfg["output"]["prefix"].rstrip("/"))


# ── status / log streaming ────────────────────────────────────────────────────

def _stream_logs(s3, bucket: str, job_name: str, instance_id: str = None, ec2=None,
                 region: str = None, config_path: str = None, output_prefix: str = None):
    log_key = f"jobs/{job_name}/training.log"
    status_key = f"jobs/{job_name}/status.txt"
    printed = 0

    print(f"\n=== Streaming logs for {job_name} (Ctrl-C to detach) ===\n")
    while True:
        # Print new log content
        try:
            resp = s3.get_object(Bucket=bucket, Key=log_key)
            content = resp["Body"].read().decode(errors="replace")
            if len(content) > printed:
                print(content[printed:], end="", flush=True)
                printed = len(content)
        except s3.exceptions.NoSuchKey:
            print(".", end="", flush=True)

        # Check completion
        try:
            resp = s3.get_object(Bucket=bucket, Key=status_key)
            status = resp["Body"].read().decode().strip()
            print(f"\n\n=== Job finished: {status} ===")
            if status.startswith("SUCCESS") and output_prefix:
                print(f"\nArtifacts at:  s3://{bucket}/{output_prefix}/{job_name}/")
            cfg_flag = f"--config {config_path}" if config_path else "--config <config.yaml>"
            print(f"Download with: python train_cli.py download {cfg_flag}")
            print("\nNOTE: The instance will self-terminate after uploading its final log.")
            print("      Artifacts are persisted in S3 and can be downloaded at any time.")
            # Do NOT terminate the instance here — let the user download first.
            return
        except Exception:
            pass

        time.sleep(30)


def cmd_status(args):
    cfg = load_config(args.config)
    bucket = args.bucket or cfg["data"]["bucket"]
    job_name = cfg["job_name"]
    region = get_region(args.region)
    s3 = boto3.client("s3", region_name=region)
    _stream_logs(s3, bucket, job_name,
                 config_path=args.config,
                 output_prefix=cfg["output"]["prefix"].rstrip("/"))


# ── download ──────────────────────────────────────────────────────────────────

# Files that must be present for a complete, usable artifact set
REQUIRED_ARTIFACTS = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "classifier_head.pt",
    "model_meta.json",
    "eval_results.json",
    "classification_report.txt",
    "classification_report.json",
]


def cmd_download(args):
    cfg = load_config(args.config)
    bucket = args.bucket or cfg["data"]["bucket"]
    job_name = cfg["job_name"]
    region = get_region(args.region)
    prefix = cfg["output"]["prefix"].rstrip("/")
    s3_prefix = f"{prefix}/{job_name}"
    local_out = Path(args.output or f"./trained_models/{job_name}")
    local_out.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix + "/")
    objects = {obj["Key"]: obj["Size"] for page in pages for obj in page.get("Contents", [])}

    if not objects:
        sys.exit(f"No objects found at s3://{bucket}/{s3_prefix}/\n"
                 "Training may not have completed yet.")

    print(f"Downloading {len(objects)} files from s3://{bucket}/{s3_prefix}/  →  {local_out}/\n")
    downloaded = []
    for key, s3_size in sorted(objects.items()):
        rel = key[len(s3_prefix) + 1:]
        dest = local_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        local_size = dest.stat().st_size
        match = "OK" if local_size == s3_size else "SIZE MISMATCH"
        print(f"  [{match}]  {rel:50s}  {local_size:>12,} bytes")
        downloaded.append((rel, local_size == s3_size))

    # Verify required artifacts
    print("\n=== Artifact verification ===")
    missing = []
    for name in REQUIRED_ARTIFACTS:
        path = local_out / name
        if path.exists():
            print(f"  PRESENT  {name}")
        else:
            print(f"  MISSING  {name}  ← REQUIRED")
            missing.append(name)

    if missing:
        print(f"\nWARNING: {len(missing)} required artifact(s) missing: {missing}")
    else:
        print(f"\nAll required artifacts present in {local_out}/")

    # Print eval summary
    eval_path = local_out / "eval_results.json"
    if eval_path.exists():
        results = json.loads(eval_path.read_text())
        val_acc  = results.get("val",  {}).get("eval_accuracy", "—")
        test_acc = results.get("test", {}).get("test_accuracy", "—")
        print(f"\nVal  accuracy:  {val_acc:.4f}" if isinstance(val_acc, float) else f"\nVal  accuracy:  {val_acc}")
        print(f"Test accuracy:  {test_acc:.4f}" if isinstance(test_acc, float) else f"Test accuracy:  {test_acc}")

    # Show class labels
    meta_path = local_out / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"\nClasses ({meta['num_classes']}): {meta['class_names']}")

    if not missing:
        print(f"\nDownload complete. Safe to terminate the instance if still running.")
        print(f"  aws ec2 terminate-instances --instance-ids <instance-id>")


# ── CLI parser ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="train_cli.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", required=True, help="Path to YAML config")
    shared.add_argument("--bucket", default=None, help="S3 bucket (overrides config)")
    shared.add_argument("--region", default=None, help="AWS region (overrides env)")

    # upload
    p = sub.add_parser("upload", parents=[shared], help="Sync local dataset to S3")
    p.add_argument("--dataset", required=True, help="Local dataset root (e.g. Dataset/Final_dataset)")

    # launch
    p = sub.add_parser("launch", parents=[shared], help="Launch EC2 training job")
    p.add_argument("--instance-type", default=None, help="EC2 instance type (default: g5.xlarge)")
    p.add_argument("--ami", default=None, help="Override AMI ID")
    p.add_argument("--subnet", default=None, help="Subnet ID (default: first default-VPC subnet)")
    p.add_argument("--no-wait", action="store_true", help="Return immediately; don't stream logs")
    pricing = p.add_mutually_exclusive_group()
    pricing.add_argument("--spot", action="store_true", default=True, help="Use spot instance (default, cheapest)")
    pricing.add_argument("--on-demand", action="store_true", help="Use on-demand instance (no interruptions)")

    # status
    p = sub.add_parser("status", parents=[shared], help="Stream logs for a running job")

    # download
    p = sub.add_parser("download", parents=[shared], help="Download trained model from S3")
    p.add_argument("--output", default=None, help="Local output directory")

    args = parser.parse_args()
    {"upload": cmd_upload, "launch": cmd_launch, "status": cmd_status, "download": cmd_download}[args.cmd](args)


if __name__ == "__main__":
    main()
