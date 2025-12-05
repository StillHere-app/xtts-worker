# ============================================================
#  DEBUG WRAPPER TO CATCH IMPORT FAILURES AT STARTUP
# ============================================================
try:
    import os
    import tempfile
    import traceback
    import base64
    import json

    import requests
    from google.cloud import storage
    from runpod import serverless

    import torch
    import torchaudio
    import ffmpeg
    from TTS.api import TTS

except Exception as e:
    print("🔥 IMPORT FAILURE DURING STARTUP:", e)
    traceback.print_exc()
    raise


# ============================================================
#  PYTORCH SAFE GLOBALS FOR XTTS
# ============================================================
try:
    from torch.serialization import add_safe_globals
except ImportError:
    def add_safe_globals(objs):
        return


def register_tts_safe_globals():
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


# ============================================================
#  CONFIG
# ============================================================
GCS_BUCKET = os.getenv("GCS_BUCKET")
RETURN_AUDIO_BASE64 = True

if not GCS_BUCKET:
    print("⚠️ GCS_BUCKET environment variable not set.")

gcs_client = storage.Client() if GCS_BUCKET else None
bucket = gcs_client.bucket(GCS_BUCKET) if gcs_client else None


# ============================================================
#  LAZY XTTS MODEL LOADING
# ============================================================
xtts_model = None


def get_xtts_model():
    global xtts_model
    if xtts_model is None:
        print("🔊 Loading XTTS v2 model…")
        xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("✅ XTTS v2 model ready.")
    return xtts_model


# ============================================================
#  HELPERS
# ============================================================
def upload_to_gcs(local_path: str, gcs_path: str) -> str:
    if not bucket:
        raise RuntimeError("GCS bucket is not configured (missing GCS_BUCKET).")

    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    blob.make_public()
    return blob.public_url


def download_file(url: str) -> str:
    """Download a file to a temp wav and return its path."""
    r = requests.get(url)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_embedding(sample_paths):
    print(f"📌 Extracting embedding from {len(sample_paths)} sample(s)…")
    model = get_xtts_model()
    spk_emb, gpt_latent = model.get_conditioning_latents(sample_paths)

    return {
        "speaker_embedding": spk_emb.cpu().numpy().tolist(),
        "gpt_latent": gpt_latent.cpu().numpy().tolist(),
    }


def xtts_generate_audio(text, embedding, language="en"):
    model = get_xtts_model()

    speaker = torch.tensor(embedding["speaker_embedding"])
    latent = torch.tensor(embedding["gpt_latent"])

    print("🎤 Running XTTS inference…")
    audio = model.tts_with_latents(
        text=text,
        speaker_latents=(speaker, latent),
        language=language,
    )
    return audio


def save_mp3(audio_array):
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")

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

        # ------------------------------
        # 1️⃣  EXTRACT EMBEDDING
        # ------------------------------
        if cmd == "extract-embedding":
            sample_urls = _input.get("sampleUrls") or []
            if not sample_urls:
                return {"status": "error", "message": "No sampleUrls provided."}

            sample_paths = [download_file(u) for u in sample_urls]
            embedding = extract_embedding(sample_paths)

            return {"status": "success", "embedding": embedding}

        # ------------------------------
        # 2️⃣  TTS GENERATION
        # ------------------------------
        if cmd == "tts":
            text = _input.get("text")
            embedding = _input.get("embedding")
            language = _input.get("language", "en")

            if not text or not embedding:
                return {
                    "status": "error",
                    "message": "'text' and 'embedding' are required.",
                }

            audio_array = xtts_generate_audio(text, embedding, language)
            mp3_path = save_mp3(audio_array)

            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            url = upload_to_gcs(mp3_path, gcs_path)

            return {
                "status": "success",
                "audioUrl": url,
                "audioBase64": encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None,
            }

        # ------------------------------
        # UNKNOWN COMMAND
        # ------------------------------
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
#  START SERVERLESS LOOP
# ============================================================
serverless.start({"handler": handler})
