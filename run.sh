#!/usr/bin/env bash
# Launch the NileTTS web server. Assumes ./venv exists (created by setup).
set -e
cd "$(dirname "$0")"

VENV="./venv"
if [ ! -d "$VENV" ]; then
  echo "venv not found. Create it and install deps first (see README.md)."
  exit 1
fi

source "$VENV/bin/activate"

# CTranslate2 (faster-whisper) needs cuDNN/cuBLAS at runtime; torch ships them
# inside the venv under nvidia/*/lib. Add those dirs to the loader path.
NV_LIBS=$(find "$VENV"/lib/python*/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"

# Xet transfers stall on this network — force classic HTTPS HuggingFace downloads.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

# RTX 4090 tuning: allow all memory, use expandable segments to avoid fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Set NILE_DEEPSPEED=1 to enable DeepSpeed-accelerated GPT inference (if installed).
export NILE_DEEPSPEED=${NILE_DEEPSPEED:-0}

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

echo "Starting NileTTS on http://$HOST:$PORT"
# Single worker: the model lives in GPU memory once and is shared across requests.
exec uvicorn app.server:app --host "$HOST" --port "$PORT" --workers 1
