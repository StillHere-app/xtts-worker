import os
import json
import tempfile
import traceback
import base64
import requests

from google.cloud import storage
from runpod import serverless

# ============================================================
#  PYTORCH 2.6 — SAFE GLOBALS FOR XTTS CHECKPOINT UNPICKLING
# ============================================================
import torch
from torch.serialization import add_safe_globals

#
# These are the classes that:
#  - We KNOW exist in your installed Coqui TTS version
#  - Have appeared as "unsupported global" in earlier errors
#  - Or are clearly used by the XTTS model/vocoder
#

# --- XTTS CONFIG CLASSES (CONFIGS USED BY THE MODEL) ---
from TTS.tts.configs.xtts_config import (
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
)

# --- SHARED CONFIG CLASSES (REFERENCED BY XTTS CONFIGS) ---
from TTS.config.shared_configs import (
    BaseDatasetConfig,
    AudioConfig,
    BaseTTSConfig,
)

# --- XTTS MODEL CLASSES (MODEL + SPEAKER ENCODER) ---
try:
    from TTS.tts.models.xtts import (
        Xtts,
        XttsSpeakerEncoder,
    )
except Exception:
    # If these imports ever fail in a future version,
    # we simply won't register them as safe globals.
    Xtts = None
    XttsSpeakerEncoder = None

# --- VOCODER CLASSES USED BY XTTS (FROM DISCOVERY LOGS) ---
from TTS.vocoder.models.univnet_discriminator import (
    UnivnetDiscriminator,
    TorchSTFT,
)

# --- MODEL MANAGER (OFTEN REFERENCED IN CHECKPOINT METADATA) ---
from TTS.api import ModelManager


safe_objects = [
    # Configs
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,

    BaseDatasetConfig,
    AudioConfig,
    BaseTTSConfig,

    # XTTS core classes (if present)
    Xtts,
    XttsSpeakerEncoder,

    # Vocoder bits
    UnivnetDiscriminator,
    TorchSTFT,

    # Model manager
    ModelManager,
]

# Filter out any Nones (in case some optional imports failed)
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
RETURN_AUDIO_BASE64 = True   # For previews / debugging


# ============================================================
#  LOAD XTTS MODEL (MULTILINGUAL)
# ============================================================
print("🔊 Loading XTTS v2 model (multilingual)...")
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✅ XTTS v2 model loaded successfully")


# ============================================================
#  GCS INIT
# ============================================================
gcs_client = storage.Client()
bucket = gcs_client.bucket(GCS_BUCKET)

def upload_to_gcs(local_path, gcs_path: str) -> str:
    """Upload file to GCS and return public URL."""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


# ============================================================
#  HELPERS
# ============================================================
def download_file(url: str) -> str:
    """Download file from URL into a temp .wav file, return local path."""
    print(f"⬇️ Downloading sample: {url}")
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_embedding(sample_paths):
    """
    Extract XTTS speaker embedding + GPT latent from one or more sample paths.
    """
    print(f"🧬 Extracting embedding from {len(sample_paths)} samples...")
    spk_emb, gpt_latent = xtts_model.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist(),
    }


def xtts_generate_audio(text: str, embedding: dict, language: str = "en"):
    """
    Generate XTTS audio using previously extracted embedding and a language code.
    language: e.g. "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "zh", ...
    """
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
    """
    Save XTTS numpy audio → WAV → MP3 via FFmpeg.
    """
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")

    # audio_array is 1D; torchaudio expects [channels, samples]
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

        # ----------------------------------------------------
        # HEALTH CHECK
        # ----------------------------------------------------
        if cmd == "health":
            return {
                "status": "success",
                "message": "XTTS worker alive. Model loaded.",
            }

        # ----------------------------------------------------
        # 1️⃣ EXTRACT EMBEDDING
        # ----------------------------------------------------
        if cmd == "extract-embedding":
            sample_urls = event["input"]["sampleUrls"]
            sample_paths = [download_file(u) for u in sample_urls]

            embedding = extract_embedding(sample_paths)

            return {
                "status": "success",
                "embedding": embedding,
            }

        # ----------------------------------------------------
        # 2️⃣ TTS GENERATION (MULTILINGUAL)
        # ----------------------------------------------------
        if cmd == "tts":
            text = event["input"]["text"]
            embedding = event["input"]["embedding"]
            language = event["input"].get("language", "en")

            audio_array = xtts_generate_audio(text, embedding, language=language)
            mp3_path = save_mp3(audio_array)

            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            audio_url = upload_to_gcs(mp3_path, gcs_path)

            return {
                "status": "success",
                "audioUrl": audio_url,
                "audioBase64": encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None,
            }

        # ----------------------------------------------------
        # UNKNOWN COMMAND
        # ----------------------------------------------------
        return {
            "status": "error",
            "message": f"Unknown command: {cmd}",
        }

    except Exception as e:
        print("❌ Exception in handler:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
        }


# ============================================================
#  START SERVERLESS WORKER
# ============================================================
serverless.start({"handler": handler})

