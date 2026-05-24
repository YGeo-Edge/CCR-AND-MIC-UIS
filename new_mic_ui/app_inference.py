#!/usr/bin/env python3
"""
MIC2 Malicious-Ad Classifier — Inference UI
Upload individual images or an entire folder and get label + confidence for each.

Run:
    python app_inference.py
    python app_inference.py --adapter trained_models/mic2_internvl_v1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# ── label colours (hex) ───────────────────────────────────────────────────────
LABEL_COLORS = {
    "Benign":               "#22c55e",
    "fake_av":              "#ef4444",
    "financial_scam":       "#f97316",
    "misleading_offers":    "#eab308",
    "fake_appstore":        "#ec4899",
    "tech_support_scam":    "#8b5cf6",
    "gift_card_scan":       "#f43f5e",
    "forced_notification":  "#06b6d4",
    "suspicious_vpn":       "#3b82f6",
    "malicious_extension":  "#a855f7",
    "fake_updates":         "#14b8a6",
    "blank_LP":             "#64748b",
    "fake_downloader":      "#dc2626",
}
DEFAULT_COLOR = "#6b7280"

# ── device ────────────────────────────────────────────────────────────────────
def _device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ── transform (must match training) ──────────────────────────────────────────
def _transform(image_size=448):
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ── model (loaded once at startup) ───────────────────────────────────────────
_model      = None
_class_names = None
_tf          = None
_dev         = None


def load_model(adapter_path: str):
    global _model, _class_names, _tf, _dev
    sys.path.insert(0, str(HERE))
    from model import InternVLClassifier

    _dev = _device()
    print(f"Loading model on {_dev} …")
    t0 = time.time()
    _model = InternVLClassifier.load_adapter(adapter_path, dtype=torch.float32)
    _model = _model.to(_dev).eval()
    print(f"Model ready in {time.time()-t0:.1f}s")

    with open(Path(adapter_path) / "model_meta.json") as f:
        meta = json.load(f)
    _class_names = meta["class_names"]
    _tf = _transform(448)


# ── inference ─────────────────────────────────────────────────────────────────
def _predict_one(img: Image.Image):
    x = _tf(img).unsqueeze(0).to(_dev)
    with torch.no_grad():
        logits = _model(pixel_values=x).logits
    probs     = F.softmax(logits, dim=-1)[0]
    idx       = probs.argmax().item()
    label     = _class_names[idx]
    conf      = probs[idx].item()
    top5      = [(  _class_names[i], probs[i].item())
                 for i in probs.topk(min(5, len(_class_names))).indices.tolist()]
    return label, conf, top5


# ── resolve file path (Gradio 6 passes FileData; older versions pass str/obj) ─
def _resolve_path(f) -> Path | None:
    try:
        # Gradio 6: FileData object with .path attribute
        if hasattr(f, "path"):
            return Path(f.path)
        # Plain string path
        if isinstance(f, str):
            return Path(f)
        # Older Gradio: file-like object with .name
        if hasattr(f, "name"):
            return Path(f.name)
    except Exception:
        pass
    return None


# ── Gradio handler ────────────────────────────────────────────────────────────
def classify(files):
    if not files:
        return [], "<p style='color:#6b7280'>No images uploaded.</p>"

    gallery_items = []
    rows_html     = []
    errors        = []

    for f in files:
        path = _resolve_path(f)
        if path is None or not path.exists():
            errors.append(f"Could not resolve: {f}")
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            continue

        try:
            label, conf, top5 = _predict_one(img)
        except Exception as e:
            errors.append(f"{path.name}: inference failed — {e}")
            continue

        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        # Gallery entry: (PIL image, caption)
        caption = f"{label}  {conf:.1%}"
        gallery_items.append((img, caption))

        # Detail card
        top5_bars = ""
        for lbl, p in top5:
            c   = LABEL_COLORS.get(lbl, DEFAULT_COLOR)
            pct = f"{p:.1%}"
            w   = f"{p*100:.1f}%"
            top5_bars += f"""
            <div style="margin:3px 0;display:flex;align-items:center;gap:8px;">
              <span style="width:160px;font-size:12px;color:#e2e8f0;white-space:nowrap;
                           overflow:hidden;text-overflow:ellipsis">{lbl}</span>
              <div style="flex:1;background:#334155;border-radius:4px;height:10px;">
                <div style="width:{w};background:{c};border-radius:4px;height:10px;"></div>
              </div>
              <span style="width:42px;font-size:12px;color:#94a3b8;text-align:right">{pct}</span>
            </div>"""

        rows_html.append(f"""
        <div style="background:#1e293b;border-radius:12px;padding:14px;margin:8px;
                    border-left:4px solid {color};min-width:280px;max-width:340px;
                    display:inline-block;vertical-align:top;">
          <p style="margin:0 0 4px 0;font-size:13px;color:#94a3b8;
                    word-break:break-all;">{path.name}</p>
          <p style="margin:0 0 10px 0;font-size:18px;font-weight:700;color:{color};">
            {label}
          </p>
          <p style="margin:0 0 10px 0;font-size:26px;font-weight:800;color:#f8fafc;">
            {conf:.1%}
          </p>
          <div>{top5_bars}</div>
        </div>""")

    if not gallery_items:
        err_txt = "<br>".join(errors) if errors else "No valid images found."
        return [], f"<p style='color:#ef4444'>{err_txt}</p>"

    error_section = ""
    if errors:
        error_section = f"""
        <p style="color:#f97316;font-size:12px;margin:8px 0 0 0;">
          {len(errors)} file(s) skipped: {"; ".join(errors[:3])}
        </p>"""

    detail_html = f"""
    <div style="background:#0f172a;padding:16px;border-radius:12px;">
      <p style="color:#94a3b8;font-size:13px;margin:0 0 12px 0;">
        {len(gallery_items)} image(s) classified
      </p>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        {"".join(rows_html)}
      </div>
      {error_section}
    </div>"""

    return gallery_items, detail_html


# ── UI ────────────────────────────────────────────────────────────────────────
_THEME = gr.themes.Base(
    primary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    block_background_fill="#1e293b",
    block_border_color="#334155",
    input_background_fill="#1e293b",
)

_CSS = """
.gradio-container { max-width: 1400px !important; }
#title { text-align:center; padding: 24px 0 8px 0; }
#title h1 { color:#f8fafc; font-size:28px; margin:0; }
#title p  { color:#94a3b8; font-size:14px; margin:4px 0 0 0; }
"""


def build_ui():
    with gr.Blocks(title="MIC2 Ad Classifier") as demo:

        with gr.Column(elem_id="title"):
            gr.HTML("""
            <h1>MIC2 Malicious-Ad Classifier</h1>
            <p>Upload screenshots — get label + confidence for each image instantly.</p>
            """)

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                upload = gr.File(
                    label="Upload images or drag-drop a folder",
                    file_count="multiple",
                    file_types=["image"],
                )
                run_btn = gr.Button("Classify", variant="primary", size="lg")
                gr.HTML("""
                <div style="margin-top:12px;padding:12px;background:#0f172a;
                            border-radius:8px;font-size:12px;color:#64748b;line-height:1.6">
                  <b style="color:#94a3b8">Supported formats:</b><br>
                  JPG · PNG · WebP · BMP<br><br>
                  <b style="color:#94a3b8">Tip — upload a folder:</b><br>
                  Drag the entire folder onto the upload box
                  or use <em>Select Files</em> and ⌘-A / Ctrl-A to select all.
                </div>
                """)

            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Results",
                    columns=4,
                    height=480,
                    object_fit="contain",
                    show_label=True,
                    elem_id="results-gallery",
                )

        detail = gr.HTML(label="Details")

        run_btn.click(fn=classify, inputs=upload, outputs=[gallery, detail])

    return demo


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=str(HERE / "trained_models/mic2_internvl_v1"),
        help="Path to adapter directory",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    load_model(args.adapter)
    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True,
              theme=_THEME, css=_CSS)


if __name__ == "__main__":
    main()
