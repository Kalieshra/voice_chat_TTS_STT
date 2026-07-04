"""
Speech-to-Text engine — faster-whisper (large-v3) tuned for Egyptian Arabic.

Pipeline (transcribe):
    audio bytes → [ffmpeg clean] → [Whisper] → [normalize text] → text

Design notes:
  * Model is loaded once and cached (`_MODEL_CACHE`); loading is lazy and
    lock-protected so two concurrent requests can't load it twice.
  * GPU path: device="cuda", compute_type="float16". CPU fallback: int8, 16 threads.
  * ffmpeg is resolved without root via imageio-ffmpeg's bundled static binary
    (falls back to a system ffmpeg if present).
  * Egyptian tuning = initial_prompt + hotwords + post-normalization.
  * Concurrency is capped with an asyncio.Semaphore(2) to protect VRAM; the heavy
    synchronous work runs in asyncio.to_thread so the event loop stays free.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import threading
import time

# --- ffmpeg resolution (no root needed) ------------------------------------
def _resolve_ffmpeg() -> str:
    from shutil import which

    sys_ffmpeg = which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "No ffmpeg found and imageio-ffmpeg unavailable. "
            "pip install imageio-ffmpeg"
        ) from e


FFMPEG = None  # resolved lazily


def _ffmpeg() -> str:
    global FFMPEG
    if FFMPEG is None:
        FFMPEG = _resolve_ffmpeg()
    return FFMPEG


# --- model cache ------------------------------------------------------------
_MODEL_CACHE: dict = {}
_LOAD_LOCK = threading.Lock()
_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")


def get_model():
    """Lazy, lock-protected, cached WhisperModel loader."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]
    with _LOAD_LOCK:
        if "model" in _MODEL_CACHE:  # double-checked
            return _MODEL_CACHE["model"]

        # Xet transfers stall on this network; force classic HTTPS downloads.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from faster_whisper import WhisperModel

        try:
            import torch

            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False

        t0 = time.time()
        if has_cuda:
            model = WhisperModel(_MODEL_SIZE, device="cuda", compute_type="float16")
            dev = "cuda/float16"
        else:
            model = WhisperModel(
                _MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=16
            )
            dev = "cpu/int8"
        print(f"[STT] loaded {_MODEL_SIZE} on {dev} in {time.time() - t0:.1f}s")
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["device"] = dev
        return model


# --- audio cleanup (ffmpeg) -------------------------------------------------
def convert_audio_to_wav(raw: bytes, in_suffix: str = ".webm") -> str:
    """Convert arbitrary browser audio (webm/ogg/mp4/wav) to a cleaned 16kHz mono
    WAV: highpass/lowpass band-limit, afftdn denoise, loudnorm volume. Returns the
    path to a temp WAV (caller deletes it)."""
    in_fd, in_path = tempfile.mkstemp(suffix=in_suffix)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(in_fd)
    os.close(out_fd)
    with open(in_path, "wb") as f:
        f.write(raw)

    # Gentle band-limit + light denoise + loudness normalize. Kept deliberately
    # soft: aggressive denoise strips quiet consonants and hurts recognition of
    # soft/short speech. Wider top end (9kHz) preserves Arabic sibilants (س ش ص).
    filters = "highpass=f=70,lowpass=f=9000,afftdn=nf=-20,loudnorm=I=-16:TP=-1.5:LRA=11"
    cmd = [
        _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-ac", "1", "-ar", "16000",
        "-af", filters,
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Retry without filters in case the denoise chain rejects odd input.
        cmd_simple = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
            "-i", in_path, "-ac", "1", "-ar", "16000", out_path,
        ]
        try:
            subprocess.run(cmd_simple, check=True, capture_output=True)
        except subprocess.CalledProcessError as e2:
            _safe_unlink(in_path)
            _safe_unlink(out_path)
            raise RuntimeError(
                f"ffmpeg failed: {e2.stderr.decode(errors='replace')[:300]}"
            ) from e2
    finally:
        _safe_unlink(in_path)
    return out_path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# --- Egyptian Arabic tuning -------------------------------------------------
# Primes Whisper toward colloquial Egyptian style instead of MSA.
EGYPTIAN_INITIAL_PROMPT = (
    "ده كلام باللهجة المصرية العامية. "
    "إزيك عامل إيه؟ أنا كويس الحمد لله. "
    "احنا بنتكلم مصري مش فصحى."
)

# Egyptian pronouns, question words, and everyday slang to boost recognition.
EGYPTIAN_HOTWORDS = (
    "إزيك إزاي إيه ليه فين إمتى مين عايز عاوز عايزة مش كده كدا "
    "دلوقتي النهارده امبارح بكرة خلاص يلا يالا معلش طب طيب بص بصي "
    "احنا انتوا هما بتاع بتاعت علشان عشان لسه اهو اهي دا دي دول "
    "حاجة حاجات شوية كتير أوي قوي جامد تمام ماشي حلو وحش زفت "
    "بقى يعني أصل بالظبط بجد والله ربنا يخليك متشكر شكرا "
    "اقعد قوم تعالى روح هات خد اديني وريني افتكرت نسيت عارف مش عارف "
    "بتعمل إيه عملت إيه هتعمل إيه فيه إيه مالك مالو "
    "الأوضة الشباك الموبايل العربية الشغل البيت المدرسة"
)


def normalize_egyptian_text(text: str) -> str:
    """Post-process Whisper output toward Egyptian Arabic + clean hallucinations."""
    if not text:
        return ""
    text = text.strip()

    # Common MSA -> Egyptian lexical swaps (whole-word, spaced).
    swaps = {
        "ماذا": "إيه",
        "كيف": "إزاي",
        "لماذا": "ليه",
        "أين": "فين",
        "متى": "إمتى",
        "من": "مين",  # note: risky, only when standalone question — applied word-boundary
        "الآن": "دلوقتي",
        "الآنَ": "دلوقتي",
        "اليوم": "النهارده",
        "نعم": "أيوة",
        "جيد": "كويس",
        "كثيرا": "كتير",
        "قليلا": "شوية",
        "أريد": "عايز",
        "لست": "مش",
        "ليس": "مش",
        "هكذا": "كده",
        "حسنا": "طب",
    }
    for msa, egy in swaps.items():
        text = re.sub(rf"(?<!\S){re.escape(msa)}(?!\S)", egy, text)

    # Strip tashkeel (diacritics) and the superscript alef.
    text = re.sub(r"[ً-ٰٟ]", "", text)
    # Remove tatweel.
    text = text.replace("ـ", "")
    # Normalize alef variants and ya/alef-maqsura.
    text = re.sub(r"[إأآ]", "ا", text)
    text = text.replace("ى", "ي")

    # Collapse immediate word repetitions (Whisper stutter/hallucination).
    text = re.sub(r"\b(\S+)(\s+\1\b){1,}", r"\1", text)
    # Collapse repeated punctuation and stray runs of dots/commas.
    text = re.sub(r"([.،!?])\1{1,}", r"\1", text)
    # Squeeze whitespace.
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# --- transcription ----------------------------------------------------------
def transcribe(raw_audio: bytes, in_suffix: str = ".webm") -> dict:
    """Full synchronous pipeline. Returns {text, language, language_probability,
    duration, rtf, device}."""
    model = get_model()
    wav_path = convert_audio_to_wav(raw_audio, in_suffix=in_suffix)
    try:
        t0 = time.time()
        segments, info = model.transcribe(
            wav_path,
            language="ar",
            beam_size=5,          # accuracy over speed (was 1)
            best_of=5,            # more candidates when temperature fallback kicks in
            # Temperature fallback: retry with a little sampling if a greedy decode
            # looks degenerate — recovers words a single pass would drop.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
            # More sensitive VAD: lower speech-probability gate catches quiet/soft
            # speech, extra padding stops it clipping the first/last word, and a
            # longer silence gate avoids chopping mid-sentence pauses.
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.3,
                min_silence_duration_ms=700,
                speech_pad_ms=400,
            ),
            initial_prompt=EGYPTIAN_INITIAL_PROMPT,
            hotwords=EGYPTIAN_HOTWORDS,
            no_speech_threshold=0.45,  # was 0.6 — accept more borderline audio as speech
            hallucination_silence_threshold=2.0,
            repetition_penalty=1.2,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        raw_text = "".join(seg.text for seg in segments)
        gen = time.time() - t0
        text = normalize_egyptian_text(raw_text)
        audio_dur = getattr(info, "duration", 0.0) or 0.0
        return {
            "text": text,
            "raw_text": raw_text.strip(),
            "language": getattr(info, "language", "ar"),
            "language_probability": round(
                getattr(info, "language_probability", 0.0), 3
            ),
            "audio_duration": round(audio_dur, 2),
            "transcribe_seconds": round(gen, 2),
            "rtf": round(gen / audio_dur, 3) if audio_dur else None,
            "device": _MODEL_CACHE.get("device", "unknown"),
        }
    finally:
        _safe_unlink(wav_path)


# --- async wrapper with concurrency cap ------------------------------------
_SEM = asyncio.Semaphore(2)


async def transcribe_async(raw_audio: bytes, in_suffix: str = ".webm") -> dict:
    async with _SEM:
        return await asyncio.to_thread(transcribe, raw_audio, in_suffix)


def warm() -> None:
    """Preload the model (used at server startup)."""
    get_model()
