# Use a known-good RunPod Python base image
FROM runpod/onnxruntime:cpu-latest

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY worker.py .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port (RunPod serverless uses)
EXPOSE 8080

# Start the worker
CMD ["python3", "worker.py"]
