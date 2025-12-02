# ============================================================
# XTTS Worker Dockerfile for RunPod Serverless
# CPU/GPU Compatible
# ============================================================

# Cache-buster to force rebuilds when needed
ARG CACHEBUSTER=1

# Base image with CUDA support (works for CPU as well)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Accept the Coqui XTTS license automatically
ENV COQUI_TOS_AGREED=1

# Disable Python buffering
ENV PYTHONUNBUFFERED=1

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
# Copy code into container
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Python Dependencies
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# Install Coqui TTS (latest stable)
RUN pip install TTS==0.20.2

# Install additional worker requirements
RUN pip install -r requirements.txt || true

# ------------------------------------------------------------
# Expose HTTP port for RunPod serverless
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Execution command
# ------------------------------------------------------------
CMD ["python3", "/app/worker.py"]
