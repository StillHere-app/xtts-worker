import os
import json
import traceback
import pkgutil
import inspect

from runpod import serverless

print("\n==============================")
print("🔍 STARTING XTTS DISCOVERY MODE")
print("==============================\n")

# ------------------------------------------------------------
# 1. Show TTS version
# ------------------------------------------------------------
try:
    import TTS
    print(f"📦 Installed Coqui TTS version: {getattr(TTS, '__version__', 'UNKNOWN')}")
except Exception as e:
    print("❌ Could not import TTS at all:", e)
    TTS = None


# ------------------------------------------------------------
# 2. MODULE SCAN — Find every TTS.* module and try to import it
# ------------------------------------------------------------
print("\n==============================")
print("📁 SCANNING MODULES")
print("==============================\n")

import_failures = []
import_successes = []

if TTS:
    for module_info in pkgutil.walk_packages(TTS.__path__, prefix="TTS."):
        name = module_info.name
        try:
            __import__(name)
            import_successes.append(name)
        except Exception as e:
            import_failures.append((name, str(e)))


print("✅ Successful module imports:", len(import_successes))
for m in import_successes:
    print("  ✔", m)

print("\n⚠️ Failed module imports:", len(import_failures))
for m, err in import_failures:
    print("  ❌", m, "→", err)


# ------------------------------------------------------------
# 3. CLASS SCAN — Find every class defined in the TTS package
# ------------------------------------------------------------
print("\n==============================")
print("🏷️ SCANNING CLASSES")
print("==============================\n")

discovered_classes = []

if TTS:
    for module_info in pkgutil.walk_packages(TTS.__path__, prefix="TTS."):
        name = module_info.name
        try:
            module = __import__(name, fromlist=[''])
            for cname, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    discovered_classes.append(f"{name}.{cname}")
        except Exception:
            pass

print(f"📌 Total classes discovered: {len(discovered_classes)}\n")

for cls in discovered_classes:
    print("  🧩", cls)


# ------------------------------------------------------------
# 4. Handler (only health check for now)
# ------------------------------------------------------------
def handler(event):
    return {
        "status": "success",
        "message": "XTTS discovery worker is running. Check logs for module/class scan output."
    }


# ------------------------------------------------------------
# 5. Start worker
# ------------------------------------------------------------
serverless.start({"handler": handler})

print("\n==============================")
print("🔍 XTTS DISCOVERY MODE INITIALIZED")
print("==============================\n")
