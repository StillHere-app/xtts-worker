# ============================================================
# XTTS Worker Dockerfile for RunPod Serverless
# GPU-enabled with CUDA 12.1
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git ffmpeg wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Working directory
# ------------------------------------------------------------
WORKDIR /app

# Copy code
COPY . /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch FIRST
RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install XTTS (this may install a too-new transformers/tokenizers)
RUN pip install TTS==0.22.0

# Install worker requirements
RUN pip install -r requirements.txt || true

# Install utilities
RUN pip install runpod google-cloud-storage requests ffmpeg-python

# ------------------------------------------------------------
# FIX 1: Force the correct transformers version
# ------------------------------------------------------------
RUN pip install transformers==4.31.0 --force-reinstall --no-deps

# ------------------------------------------------------------
# FIX 2: Force compatible tokenizers version
# ------------------------------------------------------------
RUN pip install tokenizers==0.13.3 --force-reinstall --no-deps

EXPOSE 8000

CMD ["python3", "/app/worker.py"]
