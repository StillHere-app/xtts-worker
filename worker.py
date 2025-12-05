import os
import tempfile
import traceback
import base64
import json

import requests
from google.cloud import storage
from runpod import serverless

# ============================================================
#  PYTORCH 2.6+ SAFE GLOBALS FOR XTTS (DYNAMIC, SNO HARD-CODED NAMES)
# ============================================================
import torch

try:
    from torch.serialization import add_safe_globals
except ImportError:
    # Older torch – no safe globals mechanism, just no-op
    def add_safe_globals(objs):
        return


def register_tts_safe_globals():
    """
    Dynamically scan a few key TTS modules and register all class types
    with torch.serialization.add_safe_globals so XTTS checkpoints can
    unpickle safely under PyTorch 2.6+ (weights_only=True by default).
    """
    from importlib import import_module

    module_names = [
        "TTS.tts.configs.xtts_config",
        "TTS.config.shared_configs",
        "TTS.tts.models.xtts",
        "TTS.vocoder.models",
        "TTS.vocoder.models.univnet_discriminator",
        "TTS.api",
    ]

    all_classes = []

    for mod_name in module_names:
        try:
            mod = import_module(mod_name)
        except Exception:
            continue

        for attr_name in dir(mod):
            # Skip dunder & private-ish names quickly
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(mod, attr_name)
            except Exception:
                continue

            if isinstance(attr, type):
                all_classes.append(attr)

    if all_classes:
        add_safe_globals(all_classes)
        print(f"🔐 Registered {len(all_classes)} TTS classes as safe globals.")


register_tts_safe_globals()

import torchaudio
import ffmpeg
from TTS.api import TTS

# ============================================================
#  CONFIG
# ============================================================
GCS_BUCKET = os.getenv("GCS_BUCKET")
RETURN_AUDIO_BASE64 = True  # used for previews / direct returns

if not GCS_BUCKET:
    print("⚠️ GCS_BUCKET env var is not set – uploads will fail.")

gcs_client = storage.Client()
bucket = gcs_client.bucket(GCS_BUCKET) if GCS_BUCKET else None


# ============================================================
#  LOAD XTTS MODEL (v2)
# ============================================================
print("🔊 Loading XTTS v2 model…")
xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("✅ XTTS v2 model ready.")


# ============================================================
#  HELPERS
# ============================================================
def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    if not bucket:
        raise RuntimeError("GCS bucket not configured (GCS_BUCKET env missing).")

    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def download_file(url: str) -> str:
    """Download a file to a temp .wav and return the local path."""
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_embedding(sample_paths):
    """
    Extract speaker embedding & GPT latent from one or more sample wavs.

    sample_paths: list of local .wav file paths
    returns: dict with 'speaker_embedding' and 'gpt_latent' as plain Python lists
    """
    print(f"📌 Extracting embedding from {len(sample_paths)} samples…")
    spk_emb, gpt_latent = xtts_model.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist(),
    }


def xtts_generate_audio(text, embedding, language="en"):
    """
    Generate XTTS audio using a previously extracted embedding dict.
    """
    speaker = torch.tensor(embedding["speaker_embedding"])
    latent = torch.tensor(embedding["gpt_latent"])

    print(f"🎤 Running XTTS inference… (lang={language})")
    audio = xtts_model.tts_with_latents(
        text=text,
        speaker_latents=(speaker, latent),
        language=language,
    )
    return audio


def save_mp3(audio_array):
    """
    Save XTTS numpy audio → temp wav → mp3 via FFmpeg.
    """
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")

    # audio_array is 1D np.array; torchaudio expects [channels, samples]
    waveform = torch.tensor(audio_array).unsqueeze(0)
    torchaudio.save(wav_path, waveform, 24000)

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
        _input = event.get("input") or {}
        cmd = _input.get("command")

        # ----------------------------------------------------
        # 1️⃣ EXTRACT EMBEDDING
        # ----------------------------------------------------
        if cmd == "extract-embedding":
            sample_urls = _input.get("sampleUrls") or []
            if not sample_urls:
                return {
                    "status": "error",
                    "message": "No sampleUrls provided for embedding extraction.",
                }

            sample_paths = [download_file(u) for u in sample_urls]
            embedding = extract_embedding(sample_paths)

            return {
                "status": "success",
                "embedding": embedding,
            }

        # ----------------------------------------------------
        # 2️⃣ TTS GENERATION
        # ----------------------------------------------------
        if cmd == "tts":
            text = _input.get("text")
            embedding = _input.get("embedding")
            language = _input.get("language", "en")

            if not text or not embedding:
                return {
                    "status": "error",
                    "message": "Both 'text' and 'embedding' are required for tts.",
                }

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
        print("❌ EXCEPTION in handler:", e)
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc(),
        }


# ============================================================
#  START SERVERLESS LOOP
# ============================================================
serverless.start({"handler": handler})
