FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir TTS==0.22.0
RUN pip install --no-cache-dir ffmpeg-python requests google-cloud-storage
RUN pip install --no-cache-dir transformers==4.31.0 tokenizers==0.13.3
RUN pip install --no-cache-dir runpod
WORKDIR /workspace
COPY worker.py /workspace/worker.py
CMD ["python3", "-u", "worker.py"]
