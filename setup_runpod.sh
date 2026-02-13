#!/bin/bash
# ================================================================
# RunPod Pod Setup — Room Reconstruction Demo
# ================================================================
# Deploys the full pipeline on any fresh RunPod GPU pod.
# Handles: VGGT, CAGE checkpoints, torch >=2.6, blinker fix, Gemini creds.
#
# USAGE (pick one):
#
#   Option A — SCP then run:
#     scp -P <PORT> -i ~/.ssh/id_ed25519 setup_runpod.sh root@<IP>:/workspace/
#     ssh -p <PORT> -i ~/.ssh/id_ed25519 root@<IP> "bash /workspace/setup_runpod.sh"
#
#   Option B — Web terminal (paste this):
#     curl -sSL https://raw.githubusercontent.com/rajeshkanaka/Room-Reconstruction-Demo/rajeshkanaka/init-room-recon/setup_runpod.sh | bash
#
# PREREQUISITES:
#   - RunPod pod with GPU (e.g. RTX 4090, A100)
#   - Base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel (or similar)
#   - GCP service account key at /workspace/gcloud-sa.json (for Gemini)
#     If missing, Gemini analysis is skipped gracefully.
# ================================================================
set -e

echo "============================================"
echo " Room Reconstruction — RunPod Pod Setup"
echo "============================================"

cd /workspace

# ── 1. Clone / update project ──────────────────────────────────
echo "[1/8] Cloning project..."
if [ -d "missoula" ]; then
    cd missoula && git pull && cd ..
else
    git clone -b rajeshkanaka/init-room-recon \
        https://github.com/rajeshkanaka/Room-Reconstruction-Demo.git missoula
fi

# ── 2. Clone and install VGGT ──────────────────────────────────
echo "[2/8] Setting up VGGT..."
if [ -d "vggt" ]; then
    cd vggt && git pull && cd ..
else
    git clone --depth 1 https://github.com/facebookresearch/vggt.git
fi
pip install -e vggt 2>&1 | tail -1

# ── 3. System dependencies ─────────────────────────────────────
echo "[3/8] Installing system deps..."
apt-get update -qq && apt-get install -y -qq \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 2>/dev/null || true

# ── 4. Python dependencies ─────────────────────────────────────
echo "[4/8] Installing Python deps..."
cd /workspace/missoula
# Root cause: pytorch base image has blinker 1.4 installed via apt (distutils).
# pip cannot uninstall distutils packages, so we remove it with apt first.
apt-get remove -y python3-blinker 2>/dev/null || true
pip install -r requirements.txt 2>&1 | tail -5

# ── 5. Upgrade torch to >=2.6 ─────────────────────────────────
# Root cause: OpeningDetector uses SegFormer weights loaded via torch.load().
# torch <2.6 blocks torch.load() even with weights_only=True due to CVE-2025-32434.
echo "[5/8] Upgrading torch (>=2.6 required for SegFormer)..."
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
if python -c "from packaging.version import Version; exit(0 if Version('$TORCH_VER') >= Version('2.6') else 1)" 2>/dev/null; then
    echo "  torch $TORCH_VER already >=2.6, skipping."
else
    echo "  torch $TORCH_VER < 2.6, upgrading..."
    pip install --upgrade "torch>=2.6" torchvision 2>&1 | tail -3
    python -c "import torch; print(f'  torch upgraded to {torch.__version__}')"
fi

# ── 6. Download CAGE checkpoints ───────────────────────────────
# Root cause: CAGE (NeurIPS 2025) model weights are not bundled in the repo.
# Without them, floor plan detection falls back to Hough pipeline.
echo "[6/8] Downloading CAGE checkpoints..."
CAGE_DIR=/workspace/missoula/external/cage/checkpoints
mkdir -p "$CAGE_DIR"

if [ -f "$CAGE_DIR/CAGE_stru3d_resnet50.pth" ]; then
    echo "  CAGE checkpoints already present, skipping."
else
    pip install gdown 2>&1 | tail -1
    # Download both ResNet-50 (~471MB) and SwinV2 (~2.4GB) from Google Drive
    gdown --fuzzy "https://drive.google.com/drive/folders/1jajjRamJ7SVgCWB-Tihp-ToqPsv0GmE7" \
        --folder -O "$CAGE_DIR" 2>&1 | tail -5
    # gdown nests into a subfolder — flatten if needed
    if [ -d "$CAGE_DIR/CAGE_checkpoints" ]; then
        mv "$CAGE_DIR/CAGE_checkpoints"/*.pth "$CAGE_DIR/" 2>/dev/null || true
        rm -rf "$CAGE_DIR/CAGE_checkpoints"
    fi
    echo "  CAGE checkpoints downloaded:"
    ls -lh "$CAGE_DIR"/*.pth 2>/dev/null || echo "  WARNING: No .pth files found!"
fi

# ── 7. Pre-download VGGT model (~4GB) ──────────────────────────
echo "[7/8] Pre-downloading VGGT model..."
export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
try:
    snapshot_download('facebook/VGGT-1B', cache_dir=cache)
    print('  VGGT model cached.')
except Exception as e:
    print(f'  Model pre-download skipped: {e}')
    print('  Will download on first inference instead.')
"

# ── 8. Launch Gradio app ───────────────────────────────────────
echo "[8/8] Starting Gradio app..."
export GRADIO_SHARE=true
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_CLOUD_PROJECT=adktalentpulse360
export KMP_DUPLICATE_LIB_OK=TRUE

# Gemini credentials (optional — Gemini is skipped gracefully if missing)
if [ -f /workspace/gcloud-sa.json ]; then
    export GOOGLE_APPLICATION_CREDENTIALS=/workspace/gcloud-sa.json
    echo "  GCP credentials loaded from /workspace/gcloud-sa.json"
else
    echo "  WARNING: /workspace/gcloud-sa.json not found. Gemini analysis will be skipped."
    echo "  To enable: SCP your service account key to /workspace/gcloud-sa.json"
fi

echo ""
echo "============================================"
echo " Setup complete! Starting app..."
echo " A public gradio.live URL will appear below."
echo "============================================"
echo ""
python app.py
