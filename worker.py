import os
import json
import tempfile
import traceback
import base64
import requests

from google.cloud import storage
from runpod import serverless

# ============================================================
#  PYTORCH 2.6 — SAFE GLOBALS FOR XTTS UNPICKLING
# ============================================================
import torch
from torch.serialization import add_safe_globals

#
# IMPORTANT:
# These classes are the ones CONFIRMED to exist
# in your installed Coqui TTS version based on your logs.
# We DO NOT import AudioConfig or CharactersConfig,
# because they DO NOT exist in your shared_configs module.
#

# --- XTTS CONFIG CLASSES ---
from TTS.tts.configs.xtts_config import (
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
)

# --- SHARED CONFIG CLASSES (only the ones confirmed in logs) ---
from TTS.config.shared_configs import (
    BaseDatasetConfig,
    BaseTTSConfig,
)

# --- XTTS MODEL CLASSES ---
try:
    from TTS.tts.models.xtts import (
        Xtts,
        XttsSpeakerEncoder,
    )
except Exception:
    Xtts = None
    XttsSpeakerEncoder = None

# --- VOCODER CLASSES USED BY XTTS ---
from TTS.vocoder.models.univnet_discriminator import (
    UnivnetDiscriminator,
    TorchSTFT,
)

# --- MODEL MANAGER ---
from TTS.api import ModelManager


safe_objects = [
    # XTTS configs
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,

    # Shared configs (existing only)
    BaseDatasetConfig,
    BaseTTSConfig,

    # XTTS model components
    Xtts,
    XttsSpeakerEncoder,

    # Vocoder components
    UnivnetDiscriminator,
    TorchSTFT,

    # Model Manager
    ModelManager,
]

# Remove None objects in case optional imports failed
safe_objects = [obj for obj in safe_objects if obj is not None]

add_safe_globals(safe_objects)


# ============================================================
#  OTHER IMPORTS
# ============================================================
import torchaudio
import ffmpeg
from TTS.api import TTS


# ============================================================
#  CONFIG
# ============================================================
GCS_BUCKET = os.getenv("GCS_BUCKET")
RETURN_AUDIO_BASE64 = True


# ============================================================
#  LOAD XTTS MODEL
# ============================================================
print("🔊 Loading XTTS v2 model (multilingual)...")
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✅ XTTS v2 model loaded successfully")


# ============================================================
#  GCS UPLOAD
# ============================================================
gcs_client = storage.Client()
bucket = gcs_client.bucket(GCS_BUCKET)

def upload_to_gcs(local_path, gcs_path: str) -> str:
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


# ============================================================
#  HELPERS
# ============================================================
def download_file(url: str) -> str:
    """Download file into temp WAV."""
    print(f"⬇️ Downloading sample: {url}")
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_embedding(sample_paths):
    """Extract XTTS speaker embedding + GPT latent."""
    print(f"🧬 Extracting embedding from {len(sample_paths)} samples...")
    spk_emb, gpt_latent = xtts_model.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist(),
    }


def xtts_generate_audio(text: str, embedding: dict, language: str = "en"):
    """Generate XTTS audio (multilingual)."""
    print(f"🎤 Generating XTTS audio (lang={language})...")

    speaker_emb = torch.tensor(embedding["speaker_embedding"])
    gpt_latent = torch.tensor(embedding["gpt_latent"])

    audio = xtts_model.tts_with_latents(
        text=text,
        speaker_latents=(speaker_emb, gpt_latent),
        language=language,
    )
    return audio


def save_mp3(audio_array):
    """Convert WAV → MP3."""
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")

    torchaudio.save(wav_path, torch.tensor([audio_array]), 24000)
    ffmpeg.input(wav_path).output(mp3_path).run(overwrite_output=True)

    return mp3_path


def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
#  RUNPOD HANDLER
# ============================================================
def handler(event):
    try:
        cmd = event["input"].get("command")

        # Health check
        if cmd == "health":
            return {"status": "success", "message": "XTTS worker alive. Model loaded."}

        # 1️⃣ Extract embedding
        if cmd == "extract-embedding":
            sample_urls = event["input"]["sampleUrls"]
            sample_paths = [download_file(u) for u in sample_urls]

            embedding = extract_embedding(sample_paths)

            return {"status": "success", "embedding": embedding}

        # 2️⃣ Generate audio
        if cmd == "tts":
            text = event["input"]["text"]
            embedding = event["input"]["embedding"]
            language = event["input"].get("language", "en")

            audio_array = xtts_generate_audio(text, embedding, language)
            mp3_path = save_mp3(audio_array)

            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            audio_url = upload_to_gcs(mp3_path, gcs_path)

            return {
                "status": "success",
                "audioUrl": audio_url,
                "audioBase64": encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None,
            }

        return {"status": "error", "message": f"Unknown command: {cmd}"}

    except Exception as e:
        print("❌ Exception in handler:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
        }


# ============================================================
#  START SERVERLESS
# ============================================================
serverless.start({"handler": handler})
