import os
import json
import tempfile
import traceback
import base64
import requests

from google.cloud import storage
from runpod import serverless

# ============================================================
#  UNIVERSAL XTTS UNPICKLE FIX (Pytorch 2.6+)
# ============================================================
import torch
import torchaudio
import ffmpeg

# Allow ANY class inside TTS.* during unpickling.
if hasattr(torch.serialization, "safe_globals_allow_regex"):
    torch.serialization.safe_globals_allow_regex(r"^TTS\..*")
else:
    from torch.serialization import add_safe_globals
    import TTS
    add_safe_globals([TTS])

from TTS.api import TTS


# ============================================================
#  CONFIG
# ============================================================
GCS_BUCKET = os.getenv("GCS_BUCKET")
RETURN_AUDIO_BASE64 = True


# ============================================================
#  LOAD XTTS MODEL
# ============================================================
print("🔊 Loading XTTS v2 model...")
xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✓ XTTS model ready")


# ============================================================
#  GOOGLE CLOUD STORAGE
# ============================================================
gcs_client = storage.Client()
bucket = gcs_client.bucket(GCS_BUCKET)

def upload_to_gcs(local_path, gcs_path):
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


# ============================================================
#  HELPERS
# ============================================================
def download_file(url):
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_embedding(sample_paths):
    print(f"📌 Extracting embedding from {len(sample_paths)} samples...")
    spk_emb, gpt_latent = xtts.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist()
    }


def xtts_generate(text, embedding):
    speaker_emb = torch.tensor(embedding["speaker_embedding"])
    gpt_latent = torch.tensor(embedding["gpt_latent"])

    print("🎤 Running XTTS inference...")
    audio = xtts.tts_with_latents(
        text=text,
        speaker_latents=(speaker_emb, gpt_latent),
        language="en"
    )
    return audio


def save_mp3(audio_array):
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")

    torchaudio.save(wav_path, torch.tensor([audio_array]), 24000)
    ffmpeg.input(wav_path).output(mp3_path).run(overwrite_output=True)

    return mp3_path


def encode_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
#  RUNPOD HANDLER
# ============================================================
def handler(event):
    try:
        cmd = event["input"].get("command")

        # ----------------------------------------------------
        # 1️⃣ EXTRACT EMBEDDING
        # ----------------------------------------------------
        if cmd == "extract-embedding":
            sample_urls = event["input"]["sampleUrls"]
            sample_paths = [download_file(u) for u in sample_urls]

            embedding = extract_embedding(sample_paths)

            return {
                "status": "success",
                "embedding": embedding
            }

        # ----------------------------------------------------
        # 2️⃣ TTS GENERATION
        # ----------------------------------------------------
        if cmd == "tts":
            text = event["input"]["text"]
            embedding = event["input"]["embedding"]

            audio_array = xtts_generate(text, embedding)
            mp3_path = save_mp3(audio_array)

            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            audio_url = upload_to_gcs(mp3_path, gcs_path)

            return {
                "status": "success",
                "audioUrl": audio_url,
                "audioBase64": encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None
            }

        return {
            "status": "error",
            "message": f"Unknown command: {cmd}"
        }

    except Exception as e:
        print("❌ EXCEPTION:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }


# ============================================================
#  START SERVERLESS HANDLER
# ============================================================
serverless.start({"handler": handler})
