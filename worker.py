# ============================================================
#  DEBUG WRAPPER TO CATCH IMPORT FAILURES AT STARTUP
# ============================================================
try:
    import os
    import gc
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
#  PYTORCH SAFE GLOBALS FOR XTTS (PyTorch 2.x compatibility)
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
DOWNLOAD_TIMEOUT = 120  # seconds
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
        if torch.cuda.is_available():
            print(f"🖥️ Moving model to GPU: {torch.cuda.get_device_name(0)}")
            xtts_model = xtts_model.to("cuda")
        print("✅ XTTS v2 model ready.")
    return xtts_model
# ============================================================
#  GPU MEMORY MANAGEMENT
# ============================================================
def clear_gpu_memory():
    """Release GPU memory after processing to prevent OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleared")
# ============================================================
#  TEMP FILE CLEANUP
# ============================================================
def cleanup_files(file_list):
    """Clean up temporary files to prevent disk space issues."""
    for f in file_list:
        try:
            if f and os.path.exists(f):
                os.remove(f)
                print(f"🗑️ Cleaned up: {f}")
        except Exception as e:
            print(f"⚠️ Failed to cleanup {f}: {e}")
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
    print(f"📥 Downloading: {url[:80]}...")
    r = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(r.content)
    tmp.close()
    print(f"✓ Downloaded {len(r.content)} bytes to {tmp.name}")
    return tmp.name
def extract_embedding(sample_paths):
    print(f"📌 Extracting embedding from {len(sample_paths)} sample(s)…")
    model = get_xtts_model()
    
    try:
        # Try the direct synthesizer access (more reliable)
        if hasattr(model, 'synthesizer') and hasattr(model.synthesizer, 'tts_model'):
            print("📌 Using synthesizer.tts_model.get_conditioning_latents")
            spk_emb, gpt_latent = model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=sample_paths,
                gpt_cond_len=6,
                gpt_cond_chunk_len=4
            )
        else:
            # Fallback for different TTS versions
            print("📌 Using model.get_conditioning_latents (fallback)")
            spk_emb, gpt_latent = model.get_conditioning_latents(sample_paths)
        result = {
            "speaker_embedding": spk_emb.cpu().numpy().tolist(),
            "gpt_latent": gpt_latent.cpu().numpy().tolist(),
        }
        
        # Release tensors
        del spk_emb, gpt_latent
        clear_gpu_memory()
        
        return result
    except Exception as e:
        clear_gpu_memory()
        raise e
def xtts_generate_audio(text, embedding, language="en"):
    model = get_xtts_model()
    speaker = torch.tensor(embedding["speaker_embedding"])
    latent = torch.tensor(embedding["gpt_latent"])
    
    if torch.cuda.is_available():
        speaker = speaker.cuda()
        latent = latent.cuda()
    print(f"🎤 Running XTTS inference for: '{text[:50]}...'")
    
    try:
        # Try the direct synthesizer access (more reliable)
        if hasattr(model, 'synthesizer') and hasattr(model.synthesizer, 'tts_model'):
            print("🎤 Using synthesizer.tts_model.inference")
            out = model.synthesizer.tts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=latent,
                speaker_embedding=speaker,
                enable_text_splitting=True
            )
            audio = out["wav"]
        else:
            # Fallback for different TTS versions
            print("🎤 Using model.tts_with_latents (fallback)")
            audio = model.tts_with_latents(
                text=text,
                speaker_latents=(speaker, latent),
                language=language,
            )
        
        # Release tensors
        del speaker, latent
        if 'out' in dir():
            del out
        clear_gpu_memory()
        
        return audio
    except Exception as e:
        clear_gpu_memory()
        raise e
def save_mp3(audio_array):
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    mp3_path = wav_path.replace(".wav", ".mp3")
    waveform = torch.tensor(audio_array).unsqueeze(0)
    torchaudio.save(wav_path, waveform, 24000)
    ffmpeg.input(wav_path).output(mp3_path, audio_bitrate="192k").run(
        overwrite_output=True,
        quiet=True
    )
    return wav_path, mp3_path
def encode_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
# ============================================================
#  RUNPOD HANDLER
# ============================================================
def handler(event):
    temp_files = []
    
    try:
        _input = event.get("input") or {}
        cmd = _input.get("command")
        print(f"🚀 Received command: {cmd}")
        # ------------------------------
        # HEALTH CHECK
        # ------------------------------
        if cmd == "health":
            return {
                "status": "success",
                "message": "Worker is healthy",
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        # ------------------------------
        # 1️⃣  EXTRACT EMBEDDING
        # ------------------------------
        if cmd == "extract-embedding":
            sample_urls = _input.get("sampleUrls") or []
            if not sample_urls:
                return {"status": "error", "message": "No sampleUrls provided."}
            print(f"📥 Downloading {len(sample_urls)} samples...")
            sample_paths = []
            for url in sample_urls:
                path = download_file(url)
                sample_paths.append(path)
                temp_files.append(path)
            
            embedding = extract_embedding(sample_paths)
            print("✅ Embedding extracted successfully")
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
            wav_path, mp3_path = save_mp3(audio_array)
            temp_files.extend([wav_path, mp3_path])
            gcs_path = f"xtts_outputs/{os.path.basename(mp3_path)}"
            url = upload_to_gcs(mp3_path, gcs_path)
            
            audio_base64 = encode_b64(mp3_path) if RETURN_AUDIO_BASE64 else None
            print(f"✅ TTS generated: {len(audio_base64) if audio_base64 else 0} chars base64")
            return {
                "status": "success",
                "audioUrl": url,
                "audioBase64": audio_base64,
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
    
    finally:
        # Always cleanup temp files and GPU memory
        cleanup_files(temp_files)
        clear_gpu_memory()
# ============================================================
#  START SERVERLESS LOOP
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting XTTS RunPod Serverless Worker...")
    print(f"   GCS_BUCKET: {GCS_BUCKET}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    serverless.start({"handler": handler})
