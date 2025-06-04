#!/usr/bin/env bash
# Simple setup script to run the AI Video Fixer locally.
# Creates a virtual environment, installs dependencies,
# downloads lightweight model checkpoints on first run and
# launches the Gradio app.
set -e

# Determine repo root (directory of this script)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Install Python requirements
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download required models (run once)
python - <<'PY'
from demucs.pretrained import get_model
get_model('htdemucs_light')
from faster_whisper import WhisperModel
WhisperModel('tiny-int8', download_root='./models')
PY

# 4. Launch the application
python video-fixer/app.py
