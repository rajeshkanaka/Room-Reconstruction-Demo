# RunPod Serverless - Room Reconstruction
# Base: PyTorch 2.4 + CUDA 12.4 + cuDNN 9 (Python 3.11)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System deps (OpenCV, Open3D, general build tools) ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install VGGT from Facebook Research ────────────────────────────────────
RUN git clone --depth 1 https://github.com/facebookresearch/vggt.git /tmp/vggt \
    && pip install --no-cache-dir -e /tmp/vggt

# ── Python dependencies ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt runpod

# ── Copy project code ─────────────────────────────────────────────────────
COPY . .

# ── Model cache: use network volume if mounted, else local ────────────────
# Mount a RunPod network volume at /runpod-volume to persist model cache
# across cold starts. Without it, VGGT (~4GB) downloads on every cold start.
ENV HF_HOME=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Fallback: if no volume mounted, use local cache
RUN mkdir -p /root/.cache/huggingface /root/.cache/torch

# ── Gemini / Vertex AI (set these as RunPod secrets) ──────────────────────
# GOOGLE_GENAI_USE_VERTEXAI=1
# GOOGLE_CLOUD_LOCATION=global
# GOOGLE_CLOUD_PROJECT=<your-project>

# ── Entry point ───────────────────────────────────────────────────────────
CMD ["python", "-u", "rp_handler.py"]
