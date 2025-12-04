# ============================================================
# XTTS Worker Dockerfile for RunPod Serverless
# GPU-enabled with CUDA 12.1
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Non-interactive mode for apt installs
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
# Set working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy worker code
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch FIRST (CUDA 12.1 build)
RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# FIX: Install compatible HuggingFace Transformers version
# XTTS requires BeamSearchScorer which was removed in newer versions.
# ------------------------------------------------------------
RUN pip install transformers==4.31.0

# ------------------------------------------------------------
# Install XTTS (must come AFTER transformers)
# ------------------------------------------------------------
RUN pip install TTS==0.22.0

# Install your worker requirements
RUN pip install -r requirements.txt || true

# Extra utilities
RUN pip install runpod google-cloud-storage requests ffmpeg-python

# Expose HTTP port
EXPOSE 8000

# ------------------------------------------------------------
# Run worker
# ------------------------------------------------------------
CMD ["python3", "/app/worker.py"]
