# ============================================================
# Base image with Python + CUDA support (required for XTTS GPU)
# ============================================================
ENV COQUI_TOS_AGREED=1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Set work directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# Copy worker code
# ------------------------------------------------------------
COPY . /app

# ------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    runpod \
    google-cloud-storage \
    requests \
    TTS \
    ffmpeg-python

# ------------------------------------------------------------
# RunPod Serverless start command
# ------------------------------------------------------------
CMD ["python3", "worker.py"]
