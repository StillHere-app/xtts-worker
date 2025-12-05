FROM runpod/pytorch:3.0.0-py3.10-cuda12.1.1

WORKDIR /app

# -------------------------------------------------------------
# Install system dependencies
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Install Python packages
# -------------------------------------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# -------------------------------------------------------------
# Copy worker
# -------------------------------------------------------------
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

