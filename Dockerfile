# ============================================================
#  RUNPOD SERVERLESS BASE IMAGE (GPU-COMPATIBLE)
#  This is the ONLY image RunPod allows for Serverless v2.
# ============================================================
FROM runpod/serverless:latest

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
#  SYSTEM PACKAGES
# ============================================================
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
#  PYTORCH + CUDA 12.1
# ============================================================
RUN pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================
#  XTTS + DEPENDENCIES
# ============================================================
RUN pip install --no-cache-dir TTS==0.22.0
RUN pip install --no-cache-dir ffmpeg-python requests google-cloud-storage

# ============================================================
#  PINNED VERSIONS REQUIRED FOR XTTS STABILITY
# ============================================================
RUN pip install --no-cache-dir transformers==4.31.0 tokenizers==0.13.3

# ============================================================
#  COPY WORKER CODE
# ============================================================
WORKDIR /workspace
COPY worker.py /workspace/worker.py

# ============================================================
#  START WORKER
# ============================================================
CMD ["python3", "-u", "worker.py"]
