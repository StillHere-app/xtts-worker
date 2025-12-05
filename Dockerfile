# ============================================================
#  RUNPOD GPU BASE IMAGE (REQUIRED)
# ============================================================
FROM runpod/base:0.4.0

# Prevent any interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
#  SYSTEM DEPENDENCIES
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
#  PYTORCH (CUDA 12.1) — REQUIRED FOR GPU
# ============================================================
RUN pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ============================================================
#  TTS & MODEL DEPENDENCIES
# ============================================================
RUN pip install --no-cache-dir TTS==0.22.0

# Your worker depends on ffmpeg-python
RUN pip install --no-cache-dir ffmpeg-python requests google-cloud-storage

# ============================================================
#  **PIN THESE — REQUIRED FOR XTTS STABILITY**
# ============================================================
RUN pip install --no-cache-dir transformers==4.31.0 tokenizers==0.13.3

# ============================================================
#  COPY WORKER FILES
# ============================================================
WORKDIR /workspace
COPY worker.py /workspace/worker.py

# ============================================================
#  DEFAULT COMMAND
# ============================================================
CMD ["python3", "-u", "worker.py"]
