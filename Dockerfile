FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /worker

COPY requirements.txt .
RUN pip3 install -r requirements.txt --no-cache-dir

COPY worker.py .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "worker.py"]
