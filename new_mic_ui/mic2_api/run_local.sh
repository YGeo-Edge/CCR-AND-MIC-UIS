#!/bin/bash
# Run the API locally (without Docker) for development/testing
# Usage: ./run_local.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ADAPTER_PATH="../aws_training/trained_models/mic2_internvl_v1"
export BATCH_SIZE=8
export BATCH_TIMEOUT_MS=50
export JOB_TTL_S=300
export PYTHONPATH="$SCRIPT_DIR"

echo "Starting MIC2 API locally on http://localhost:8000"
echo "OpenAPI docs: http://localhost:8000/docs"
echo ""

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info \
  --reload
