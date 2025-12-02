# XTTS Worker for StillHere

This is a RunPod Serverless v2 worker that performs:

- XTTS embedding extraction
- XTTS TTS generation
- FFmpeg audio cleaning
- Google Cloud Storage upload

## Commands

### 1. extract-embedding

Extracts voice embedding from audio samples.

```json
{
  "command": "extract-embedding",
  "sampleUrls": [
    "https://storage.googleapis.com/.../sample1.wav",
    "https://storage.googleapis.com/.../sample2.wav"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "embedding": {
    "speaker_embedding": [...],
    "gpt_latent": [...]
  }
}
```

### 2. tts

Generates speech using a previously extracted embedding.

```json
{
  "command": "tts",
  "text": "Hello, this is a test message.",
  "embedding": {
    "speaker_embedding": [...],
    "gpt_latent": [...]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "audioUrl": "https://storage.googleapis.com/.../output.mp3",
  "audioBase64": "..."
}
```

## Environment Variables

- `GCS_BUCKET` - Google Cloud Storage bucket name
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCS service account JSON

## Deployment to RunPod

1. Build the Docker image:
   ```bash
   docker build -t xtts-worker .
   ```

2. Push to Docker Hub or another registry:
   ```bash
   docker tag xtts-worker your-registry/xtts-worker:latest
   docker push your-registry/xtts-worker:latest
   ```

3. Create a new RunPod Serverless endpoint with the image URL

4. Configure environment variables in RunPod dashboard

5. Note the endpoint URL and add it to StillHere as `RUNPOD_XTTS_ENDPOINT`
