"""
FastAPI server for the NileTTS Text-to-Speech system.

Endpoints:
  GET  /                     -> the web UI (text box + Speak button)
  GET  /api/voices           -> available reference voices + languages
  POST /api/tts              -> JSON {text, language, voice, ...} -> full WAV file
  POST /api/tts/stream       -> same body -> streamed 16-bit PCM (low latency)
  POST /api/upload-voice     -> upload a 6s reference .wav to clone a new voice
  GET  /healthz              -> readiness probe

Run:  ./run.sh   (or: uvicorn app.server:app --host 0.0.0.0 --port 8000)
"""

from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .tts_engine import (
    NileTTSEngine,
    SAMPLE_RATE,
    Voice,
    float_to_pcm16_bytes,
    wav_bytes,
)
from . import tts_engine
from . import stt_engine
from . import agent as agent_mod

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "nile-xtts")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_DIR = os.path.join(VOICES_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Friendly display names for the two primary responder voices. The reference
# clip that shipped as default.wav is female (~190 Hz); man.wav is a male
# reference derived from it. Any other uploaded .wav keeps its filename label.
_VOICE_LABELS = {"default": "Woman", "man": "Man"}
# Order the segmented Man/Woman toggle deterministically (Woman first = default).
_VOICE_ORDER = {"default": 0, "man": 1}
# Post-synthesis pitch/formant shift (semitones). The Man voice clones the same
# clean reference as Woman, then shifts the generated audio down to male range.
_VOICE_PITCH = {"man": -6.0}


def _discover_voices() -> dict[str, Voice]:
    """Pick up every .wav in voices/ as a selectable reference voice."""
    voices: dict[str, Voice] = {}
    if os.path.isdir(VOICES_DIR):
        names = [f for f in os.listdir(VOICES_DIR) if f.lower().endswith(".wav")]
        names.sort(key=lambda f: (_VOICE_ORDER.get(os.path.splitext(f)[0], 99), f))
        for fname in names:
            key = os.path.splitext(fname)[0]
            label = _VOICE_LABELS.get(key, key.replace("_", " ").replace("-", " ").title())
            voices[key] = Voice(
                key=key, label=label, wav_path=os.path.join(VOICES_DIR, fname),
                pitch_shift=_VOICE_PITCH.get(key, 0.0),
            )
    return voices


VOICES = _discover_voices()
DEFAULT_VOICE = "default" if "default" in VOICES else (next(iter(VOICES), None))

USE_DEEPSPEED = os.environ.get("NILE_DEEPSPEED", "0") == "1"

engine = NileTTSEngine(
    model_dir=MODEL_DIR,
    voices=VOICES,
    default_voice=DEFAULT_VOICE,
    use_deepspeed=USE_DEEPSPEED,
)

app = FastAPI(title="NileTTS — English / Arabic / Egyptian TTS")


import threading

# Expected sizes of the large model files (bytes) — used to avoid trying to load
# a checkpoint that is still downloading.
_REQUIRED_FILES = {"config.json": 1, "vocab.json": 1, "model.pth": 5_500_000_000}
_LOAD_STATE = {"status": "starting", "detail": ""}


import glob

_MODEL_PTH_TOTAL_GB = 5.61  # full model.pth size on HF


def _incomplete_gb() -> float:
    """Bytes downloaded so far for the in-progress model.pth (Xet-disabled HTTP
    download writes to a .incomplete file under .cache)."""
    best = 0
    for f in glob.glob(os.path.join(MODEL_DIR, ".cache", "**", "*.incomplete"),
                       recursive=True):
        try:
            best = max(best, os.path.getsize(f))
        except OSError:
            pass
    return best / 1e9


def _model_files_ready() -> tuple[bool, str]:
    for fname, min_size in _REQUIRED_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) < min_size:
            if fname == "model.pth":
                got = max(_incomplete_gb(),
                          os.path.getsize(path) / 1e9 if os.path.exists(path) else 0)
                pct = 100 * got / _MODEL_PTH_TOTAL_GB
                return False, f"model.pth {got:.2f} / {_MODEL_PTH_TOTAL_GB:.1f} GB ({pct:.0f}%)"
            return False, f"waiting for {fname}"
    return True, ""


def _background_load() -> None:
    if DEFAULT_VOICE is None:
        _LOAD_STATE.update(status="error", detail="no reference voice in voices/")
        print("[WARN] No reference voice found in", VOICES_DIR)
        return
    # Wait for the download to finish, then load — retrying on transient errors
    # (e.g. the file exists but is still being flushed).
    while True:
        ready, detail = _model_files_ready()
        if not ready:
            _LOAD_STATE.update(status="downloading", detail=detail)
            time.sleep(10)
            continue
        try:
            _LOAD_STATE.update(status="loading", detail="loading checkpoint into GPU")
            t0 = time.time()
            print("[NileTTS] Loading model on", "GPU" if _cuda() else "CPU", "...")
            engine.load()
            print(f"[NileTTS] Model ready in {time.time() - t0:.1f}s")
            _LOAD_STATE.update(status="ready", detail="")
            return
        except Exception as e:
            _LOAD_STATE.update(status="downloading", detail=f"retrying: {e}"[:120])
            print("[NileTTS] load failed, will retry in 15s:", e)
            time.sleep(15)


@app.on_event("startup")
def _startup() -> None:
    # Serve the UI immediately; load the model in the background so the page is
    # reachable even while the (large) checkpoint is still downloading.
    threading.Thread(target=_background_load, daemon=True).start()


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: str = "en"
    voice: str | None = None
    temperature: float = Field(0.75, ge=0.05, le=1.5)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    diacritize: bool = False


@app.get("/healthz")
def healthz() -> dict:
    progress = None
    if not engine._loaded:
        got = max(
            _incomplete_gb(),
            os.path.getsize(os.path.join(MODEL_DIR, "model.pth")) / 1e9
            if os.path.exists(os.path.join(MODEL_DIR, "model.pth"))
            else 0,
        )
        progress = round(min(got / _MODEL_PTH_TOTAL_GB, 1.0), 4)
    return {
        "ok": True,
        "model_loaded": engine._loaded,
        "status": _LOAD_STATE["status"],
        "detail": _LOAD_STATE["detail"],
        "progress": progress,          # 0..1 fraction of model.pth downloaded
        "downloaded_gb": round(_incomplete_gb() or 0, 2),
        "total_gb": _MODEL_PTH_TOTAL_GB,
        "cuda": _cuda(),
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/api/voices")
def list_voices() -> dict:
    return {
        "languages": [
            {"code": "en", "label": "English"},
            {"code": "ar", "label": "Arabic / Egyptian (العربية / المصري)"},
        ],
        "voices": [
            {"key": v.key, "label": v.label} for v in engine.voices.values()
        ],
        "default_voice": DEFAULT_VOICE,
    }


def _ensure_ready() -> None:
    if not engine._loaded:
        raise HTTPException(503, "Model is still loading, try again in a moment.")


@app.post("/api/tts")
def tts(req: TTSRequest) -> Response:
    _ensure_ready()
    try:
        t0 = time.time()
        wav = engine.synthesize(
            text=req.text,
            language=req.language,
            voice=req.voice,
            temperature=req.temperature,
            speed=req.speed,
            diacritize=req.diacritize,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if wav.size == 0:
        raise HTTPException(400, "No audio generated (empty text?).")
    data = wav_bytes(wav, SAMPLE_RATE)
    dur = wav.size / SAMPLE_RATE
    gen = time.time() - t0
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{dur:.2f}",
            "X-Gen-Seconds": f"{gen:.2f}",
            "X-RTF": f"{(gen / dur) if dur else 0:.3f}",
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )


@app.post("/api/tts/stream")
def tts_stream(req: TTSRequest) -> StreamingResponse:
    _ensure_ready()
    try:
        language = engine._validate_lang(req.language)
    except ValueError as e:
        raise HTTPException(400, str(e))

    def gen():
        for chunk in engine.stream(
            text=req.text,
            language=language,
            voice=req.voice,
            temperature=req.temperature,
            speed=req.speed,
            diacritize=req.diacritize,
        ):
            if chunk.size:
                yield float_to_pcm16_bytes(chunk)

    # Raw little-endian mono 16-bit PCM; the browser plays it via Web Audio.
    return StreamingResponse(
        gen(),
        media_type="audio/L16",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    label: str = Form("My Voice"),
) -> JSONResponse:
    _ensure_ready()
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(400, "Please upload a .wav/.mp3/.flac/.ogg audio file.")
    key = "user_" + uuid.uuid4().hex[:8]
    dest = os.path.join(UPLOAD_DIR, key + ".wav")
    raw = await file.read()

    # Normalize to 24kHz mono wav so XTTS is happy regardless of input format.
    try:
        import io
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # to mono
        if sr != SAMPLE_RATE:
            import torch
            import torchaudio

            audio = (
                torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SAMPLE_RATE
                )
                .numpy()
                .astype("float32")
            )
        sf.write(dest, audio, SAMPLE_RATE, subtype="PCM_16")
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")

    engine.register_voice(key, label, dest)
    return JSONResponse({"key": key, "label": label})


@app.post("/api/stt")
async def stt(file: UploadFile = File(...)) -> JSONResponse:
    """Speech-to-Text: upload recorded audio (webm/ogg/mp4/wav) → Egyptian Arabic
    transcription via faster-whisper."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty audio upload.")
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".webm"
    try:
        result = await stt_engine.transcribe_async(raw, in_suffix=suffix)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Transcription failed: {e}")
    return JSONResponse(result)


class AgentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    history: list[dict] | None = None
    model: str | None = None


@app.post("/api/agent")
async def agent_endpoint(req: AgentRequest) -> JSONResponse:
    """Send text to the selected chat model (OpenAI or Anthropic)."""
    import asyncio as _asyncio

    try:
        result = await _asyncio.to_thread(
            agent_mod.ask_agent, req.text, req.history, req.model
        )
    except agent_mod.AgentError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Agent failed: {e}")
    return JSONResponse(result)


@app.get("/api/models")
def models() -> dict:
    return agent_mod.available_models()


@app.get("/api/features")
def features() -> dict:
    return {
        "tts": engine._loaded,
        "stt_ready": "model" in stt_engine._MODEL_CACHE,
        "agent_configured": agent_mod.agent_available(),
        "diacritization": tts_engine.diacritization_available(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
