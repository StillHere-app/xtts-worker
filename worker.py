import os
import json
import tempfile
import traceback
import base64
import requests

from google.cloud import storage
from runpod import serverless

# ============================================================
#  PYTORCH 2.6 — COMPLETE XTTS UNPICKLE SAFE-GLOBAL ALLOW-LIST
# ============================================================
import torch
from torch.serialization import add_safe_globals

# ---- XTTS CONFIG CLASSES (ONLY ONES THAT EXIST IN THIS VERSION) ----
from TTS.tts.configs.xtts_config import (
    XttsConfig,
    XttsAudioConfig,
    XttsArgs
)

# ---- SHARED CONFIG CLASSES ----
from TTS.config.shared_configs import (
    BaseDatasetConfig,
    CharactersConfig,
    AudioConfig,
    BaseTTSConfig
)

# ---- XTTS MODEL CLASSES ----
from TTS.tts.models.xtts import (
    Xtts,
    XttsArgs,
    XttsSpeakerEncoder
)

# ---- XTTS LAYER CLASSES (SOME MAY NOT EXIST DEPENDING ON VERSION) ----
try:
    from TTS.tts.layers.xtts.transformer import XttsTransformer
    from TTS.tts.layers.xtts.audio_encoder import XttsAudioEncoder
    from TTS.tts.layers.xtts.decoder import XttsDecoder
    from TTS.tts.layers.xtts.speaker_encoder import XttsSpeakerEncoderLayer
    from TTS.tts.layers.xtts.latent_encoder import XttsLatentEncoder
except Exception:
    XttsTransformer = None
    XttsAudioEncoder = None
    XttsDecoder = None
    XttsSpeakerEncoderLayer = None
    XttsLatentEncoder = None

# ---- REGISTER ALL VERIFIED XTTS CLASSES ----
add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,

    BaseDatasetConfig,
    CharactersConfig,
    AudioConfig,
    BaseTTSConfig,

    Xtts,
    XttsArgs,
    XttsSpeakerEncoder,

    XttsTransformer,
    XttsAudioEncoder,
    XttsDecoder,
    XttsSpeakerEncoderLayer,
    XttsLatentEncoder
])


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
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
RETURN_AUDIO_BASE64 = True   # For preview responses


# ============================================================
#  LOAD XTTS MODEL
# ============================================================
print("🔊 Loading XTTS v2 model...")
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✓ XTTS model loaded successfully")


# ============================================================
#  GCS INIT
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
    spk_emb, gpt_latent = xtts_model.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist()
    }


def xtts_generate_audio(text, embedding):
    speaker_emb = torch.tensor(embedding["speaker_embedding"])
    gpt_latent = torch.tensor(embedding["gpt_latent"])

    print("🎤 Running XTTS inference...")
    audio = xtts_model.tts_with_latents(
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
#  RUNPOD SERVERLESS HANDLER
# ============================================================
def handler(event):
    try:
        cmd = event["input"].get("command")

        # 1️⃣ Extract speaker embedding
        if cmd == "extract-embedding":
            sample_urls = event["input"]["sampleUrls"]
            sample_paths = [download_file(u) for u in sample_urls]
            embedding = extract_embedding(sample_paths)

            return {
                "status": "success",
                "embedding": embedding
            }

        # 2️⃣ Synthesize audio
        if cmd == "tts":
            text = event["input"]["text"]
            embedding = event["input"]["embedding"]

            audio_array = xtts_generate_audio(text, embedding)
            mp3_path = save_mp3(audio_array)

            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            audio_url = upload_to_gcs(mp3_path, gcs_path)

            return {
                "status": "success",
                "audioUrl": audio_url,
                "audioBase64": encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None
            }

        return {"status": "error", "message": f"Unknown command: {cmd}"}

    except Exception as e:
        print("❌ Exception occurred:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }


# ============================================================
#  START WORKER
# ============================================================
serverless.start({"handler": handler})
