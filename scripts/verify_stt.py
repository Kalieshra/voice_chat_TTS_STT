#!/usr/bin/env python3
"""
Verify faster-whisper large-v3 on GPU (CUDA + float16) transcribing Arabic audio.

CTranslate2 (used by faster-whisper) dlopen()s cuDNN/cuBLAS at runtime by SONAME.
torch 2.6 ships those shared libs inside the venv under site-packages/nvidia/*/lib.
Those dirs are NOT on the default loader path, so we prepend them to LD_LIBRARY_PATH
and (if needed) re-exec this process so the dynamic loader picks them up.

Usage:
    HF_HUB_DISABLE_XET=1 python scripts/verify_stt.py
"""
import os
import sys
import glob
import time


def _nvidia_lib_dirs():
    """Return every site-packages/nvidia/*/lib dir that contains .so files."""
    dirs = []
    for site in sys.path:
        pattern = os.path.join(site, "nvidia", "*", "lib")
        for d in glob.glob(pattern):
            if glob.glob(os.path.join(d, "*.so*")):
                dirs.append(d)
    # stable, de-duplicated order
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _ensure_ld_library_path():
    """Prepend the venv nvidia lib dirs to LD_LIBRARY_PATH and re-exec once."""
    if os.environ.get("_STT_LD_BOOTSTRAPPED") == "1":
        return
    nvidia_dirs = _nvidia_lib_dirs()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    new_parts = [d for d in nvidia_dirs if d not in parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_parts + parts)
    os.environ["_STT_LD_BOOTSTRAPPED"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Re-exec so the dynamic loader sees the updated LD_LIBRARY_PATH.
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_ld_library_path()

# --- from here on, CUDA libs are resolvable ---------------------------------
from faster_whisper import WhisperModel  # noqa: E402

AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_ar.wav")

print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH", "<empty>"))
print("Audio file      =", AUDIO)

t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model loaded on cuda/float16 in %.2fs" % (time.time() - t0))

t1 = time.time()
segments, info = model.transcribe(
    AUDIO,
    language="ar",
    beam_size=5,
    temperature=0.0,
    vad_filter=True,
    condition_on_previous_text=False,
)
text = "".join(seg.text for seg in segments)
elapsed = time.time() - t1

print("=" * 60)
print("RECOGNIZED TEXT:", text.strip())
print("DETECTED LANG  :", info.language, "prob=%.4f" % info.language_probability)
print("AUDIO DURATION :", "%.2fs" % info.duration)
print("TRANSCRIBE TIME:", "%.2fs" % elapsed)
if info.duration:
    print("REALTIME FACTOR: %.2fx (audio_sec / wall_sec)" % (info.duration / elapsed))
print("=" * 60)

# --- CPU fallback: confirm it merely constructs ------------------------------
t2 = time.time()
cpu_model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=16)
print("CPU fallback WhisperModel(device=cpu, int8, cpu_threads=16) constructed OK in %.2fs" % (time.time() - t2))
