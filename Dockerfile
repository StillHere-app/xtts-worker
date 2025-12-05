FROM runpod/serverless:latest

WORKDIR /app

# -------------------------------------------------------------
# Install system dependencies
# -------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Python dependencies
# -------------------------------------------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# -------------------------------------------------------------
# Worker code
# -------------------------------------------------------------
COPY worker.py /app/worker.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
