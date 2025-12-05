# Use a supported RunPod base image
FROM runpod/serverless:cpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY worker.py .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port RunPod uses internally
EXPOSE 8080

# Start the worker
CMD ["python3", "worker.py"]
