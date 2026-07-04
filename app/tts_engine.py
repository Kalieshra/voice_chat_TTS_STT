"""
NileTTS-XTTS engine wrapper.

Loads the fine-tuned XTTS v2 model (KickItLikeShika/NileTTS-XTTS) once, keeps it
resident on the GPU, and serves synthesis requests. Optimized for an RTX 4090:

  * TF32 matmuls enabled (big speedup on Ada, negligible quality loss).
  * Conditioning latents for each reference voice are computed once and cached,
    so per-request work is only the GPT + decoder forward pass.
  * A streaming path (`stream`) yields raw float32 PCM chunks as they are
    generated, so the browser can start playing within a few hundred ms.

Supported languages: English ("en"), Arabic / Egyptian Arabic dialect ("ar").
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

# --- PyTorch 2.6+ safe-unpickling shim -------------------------------------
# XTTS checkpoints are full pickles; torch>=2.6 defaults to weights_only=True
# which refuses to load them. We register the XTTS config classes as safe and
# also fall back to forcing weights_only=False on torch.load. This is applied
# before any checkpoint is loaded.
def _install_torch_load_shim() -> None:
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig

        torch.serialization.add_safe_globals(
            [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
        )
    except Exception:
        # Older torch (no add_safe_globals) or import layout differences: the
        # weights_only kwarg below still covers us.
        pass

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load


SAMPLE_RATE = 24000
SUPPORTED_LANGUAGES = {"en", "ar"}

# Text longer than this (per language) gets split into sentence-ish chunks so
# XTTS stays within its context window and latency-to-first-audio stays low.
# These are kept UNDER XTTS's own per-language char_limits (en=250, ar=166);
# exceeding them makes XTTS warn and can truncate/degrade the audio, so we split
# first and stay in control of the sentence boundaries.
MAX_CHARS = {"en": 240, "ar": 160}

# Accuracy-tuned XTTS GPT sampling params. The low-level model.inference() uses
# its OWN signature defaults (repetition_penalty=10.0, top_p=0.85) which do not
# match this fine-tune's config.json (repetition_penalty=5.0). Passing the
# calibrated values explicitly yields more faithful, stable pronunciation with
# fewer repeated/hallucinated syllables. Applied on both full and stream paths.
GPT_INFERENCE_PARAMS = dict(
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    length_penalty=1.0,
    enable_text_splitting=False,
    # Sampling ON for natural, human-sounding prosody: greedy decoding is
    # faithful but flat/monotone. repetition_penalty above guards against the
    # occasional sampling glitch, so we get lively intonation without looping.
    do_sample=True,
)
DEFAULT_TEMPERATURE = 0.75

# Reference conditioning: use the full reference context the model was
# configured for (config.json gpt_cond_len=30) and loudness-normalize the
# reference — both improve speaker/prosody fidelity.
# NOTE: librosa_trim_db is intentionally NOT set — in this TTS version it runs
# librosa.effects.trim on the GPU-resident reference tensor and crashes
# ("can't convert cuda:0 device type tensor to numpy").
COND_PARAMS = dict(
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,
    max_ref_length=30,
    sound_norm_refs=True,
)


import re
import unicodedata

# Characters XTTS cannot pronounce and that pollute tokenization: emoji,
# pictographs, zero-width joiners/marks, control chars, and the Arabic tatweel
# (kashida ـ) elongation which is purely decorative.
_TATWEEL = "ـ"
_ZERO_WIDTH = "".join(("​", "‌", "‍", "⁠", "﻿"))
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff"
    "←-⇿⌀-⏿⬀-⯿︀-️]"
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# Symbols XTTS can't speak naturally (markdown, technical, brackets, dashes).
# Kept out of this set on purpose: sentence punctuation . , ! ? ؟ ، ؛ : and
# % & $ which XTTS's own cleaner expands into words.
_SYMBOL_RE = re.compile(r"[@#*_~^`|<>{}\[\]()=+/\\•·►▶●◦–—―\-]+")
_QUOTE_RE = re.compile(r"[\"'“”‘’«»]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Sanitize raw text (e.g. an LLM reply) before synthesis: normalize unicode,
    drop emoji/control/zero-width chars and Arabic tatweel, collapse whitespace.
    Improves pronunciation accuracy by feeding XTTS only speakable characters."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(_TATWEEL, "")
    for z in _ZERO_WIDTH:
        text = text.replace(z, "")
    text = _EMOJI_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = _QUOTE_RE.sub("", text)
    text = _SYMBOL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


_PRONUNCIATION_PATH = os.path.join(os.path.dirname(__file__), "pronunciation.json")
_pron_cache: dict = {"mtime": None, "compiled": {}}


def _load_pronunciation() -> dict:
    """Load app/pronunciation.json, recompiling only when the file changes so
    users can edit the dictionary live without a server restart."""
    try:
        mtime = os.path.getmtime(_PRONUNCIATION_PATH)
    except OSError:
        return {}
    if _pron_cache["mtime"] == mtime:
        return _pron_cache["compiled"]
    import json

    compiled: dict = {}
    try:
        with open(_PRONUNCIATION_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for lang, mapping in data.items():
            if lang.startswith("_") or not isinstance(mapping, dict):
                continue
            rules = []
            for src, repl in mapping.items():
                if not src:
                    continue
                flags = re.IGNORECASE if lang == "en" else 0
                # \b is Unicode-aware in Python 3, so this works for Arabic too.
                rules.append((re.compile(r"\b" + re.escape(src) + r"\b", flags), repl))
            compiled[lang] = rules
    except (OSError, ValueError):
        compiled = {}
    _pron_cache["mtime"] = mtime
    _pron_cache["compiled"] = compiled
    return compiled


def apply_pronunciation(text: str, lang: str) -> str:
    """Apply the user's whole-word pronunciation respellings for this language."""
    for pattern, repl in _load_pronunciation().get(lang, []):
        text = pattern.sub(repl, text)
    return text


_diacritizer = {"loaded": False, "fn": None}


def diacritize_arabic(text: str) -> str:
    """Add Arabic diacritics (tashkeel) via mishkal if installed; otherwise a
    no-op. Rule/MSA-based, so it can misvowel Egyptian colloquial — exposed only
    as an opt-in toggle. Returns text unchanged if the library is unavailable."""
    if not _diacritizer["loaded"]:
        try:
            from mishkal.tashkeel import TashkeelClass

            vocalizer = TashkeelClass()
            _diacritizer["fn"] = vocalizer.tashkeel
        except Exception:
            _diacritizer["fn"] = None
        _diacritizer["loaded"] = True
    fn = _diacritizer["fn"]
    if fn is None:
        return text
    try:
        return fn(text)
    except Exception:
        return text


def diacritization_available() -> bool:
    diacritize_arabic("")  # trigger lazy load
    return _diacritizer["fn"] is not None


def _wrap_to_limit(text: str, limit: int) -> list[str]:
    """Break an over-long span into <=limit pieces, preferring clause delimiters
    (، , ; :) then word boundaries, and only hard-cutting a single word that is
    itself longer than the limit. Avoids slicing words/syllables mid-token."""
    out: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            out.append(buf)
            buf = ""

    for clause in re.split(r"(?<=[،,;:])\s+", text):
        clause = clause.strip()
        if not clause:
            continue
        if len(buf) + len(clause) + 1 <= limit:
            buf = (buf + " " + clause).strip()
            continue
        flush()
        if len(clause) <= limit:
            buf = clause
            continue
        for word in clause.split(" "):
            if len(buf) + len(word) + 1 <= limit:
                buf = (buf + " " + word).strip()
            else:
                flush()
                while len(word) > limit:
                    out.append(word[:limit])
                    word = word[limit:]
                buf = word
    flush()
    return out


@dataclass
class Voice:
    key: str
    label: str
    wav_path: str
    # Post-synthesis pitch/formant shift in semitones. XTTS clones timbre but
    # generates its own pitch and does NOT reliably follow a low-pitched
    # reference, so a genuinely male-sounding voice is produced by shifting the
    # generated audio down (this lowers pitch AND formants together). 0 = no shift.
    pitch_shift: float = 0.0


def _polish_output(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Clean the base (female-timbre) XTTS output BEFORE any pitch shift: trim
    quiet edges, then cut a trailing low-pitched artifact — XTTS often appends a
    <150 Hz blip that sounds like a second, male voice at the end of a phrase.
    Must run in the female domain so the pitch threshold is valid for all voices."""
    if wav.size == 0:
        return wav
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=30)
    if trimmed.size >= int(0.1 * sr):
        wav = trimmed

    # Scan the last ~0.5s and drop a sustained low-pitch (<150 Hz) tail.
    tail = min(int(0.5 * sr), wav.size)
    if tail > int(0.08 * sr):
        seg = wav[-tail:]
        f0 = librosa.yin(seg, fmin=60, fmax=400, sr=sr,
                         frame_length=1024, hop_length=256)
        good = np.where(np.isfinite(f0) & (f0 >= 150.0))[0]
        if len(good):
            last_good_end = min((good[-1] + 1) * 256 + int(0.03 * sr), tail)
            cut = wav.size - tail + last_good_end
            if int(0.1 * sr) < cut < wav.size:
                wav = wav[:cut]

    fade = int(0.008 * sr)
    if wav.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wav = wav.copy()
        wav[:fade] *= ramp
        wav[-fade:] *= ramp[::-1]
    return wav


def _apply_pitch_shift(wav: np.ndarray, n_steps: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shift pitch (and formants) by n_steps semitones, preserving duration.

    Used to turn the base (female-timbre) XTTS output into a male voice. Import
    librosa lazily so the module has no hard import-time dependency on it.
    """
    if not n_steps or wav.size == 0:
        return wav
    import librosa

    shifted = librosa.effects.pitch_shift(
        y=wav.astype(np.float32), sr=sr, n_steps=float(n_steps)
    )
    return np.asarray(shifted, dtype=np.float32).reshape(-1)


class NileTTSEngine:
    def __init__(
        self,
        model_dir: str,
        voices: dict[str, Voice],
        default_voice: str,
        use_deepspeed: bool = False,
    ) -> None:
        self.model_dir = model_dir
        self.voices = voices
        self.default_voice = default_voice
        self.use_deepspeed = use_deepspeed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._latents_cache: dict[str, tuple] = {}
        # XTTS is not thread-safe for concurrent forward passes; serialize.
        self._lock = threading.Lock()
        self._loaded = False
        self.model = None
        self.config = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        _install_torch_load_shim()

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        config = XttsConfig()
        config.load_json(os.path.join(self.model_dir, "config.json"))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=self.model_dir,
            use_deepspeed=self.use_deepspeed and self.device == "cuda",
        )
        if self.device == "cuda":
            model.cuda()
        model.eval()

        self.config = config
        self.model = model
        self._loaded = True

        # Warm the default voice's latents + a tiny forward pass so the first
        # real request from the user is fast.
        self._get_latents(self.default_voice)
        try:
            self._warmup()
        except Exception:
            pass

    def _warmup(self) -> None:
        gpt_cond_latent, speaker_embedding = self._get_latents(self.default_voice)
        with torch.no_grad():
            self.model.inference(
                text="Hi.",
                language="en",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
            )

    # -- helpers ------------------------------------------------------------
    def _resolve_voice(self, voice_key: str | None) -> Voice:
        if voice_key and voice_key in self.voices:
            return self.voices[voice_key]
        return self.voices[self.default_voice]

    def _get_latents(self, voice_key: str):
        if voice_key in self._latents_cache:
            return self._latents_cache[voice_key]
        voice = self.voices[voice_key]
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=voice.wav_path,
            **COND_PARAMS,
        )
        self._latents_cache[voice_key] = (gpt_cond_latent, speaker_embedding)
        return self._latents_cache[voice_key]

    def register_voice(self, key: str, label: str, wav_path: str) -> None:
        """Add (or replace) a reference voice, e.g. a user-uploaded sample."""
        self.voices[key] = Voice(key=key, label=label, wav_path=wav_path)
        self._latents_cache.pop(key, None)

    @staticmethod
    def _split_text(text: str, lang: str) -> list[str]:
        text = normalize_text(text)
        if not text:
            return []
        limit = MAX_CHARS.get(lang, 220)
        if len(text) <= limit:
            return [text]

        # Split on sentence terminators for both scripts, keep the delimiter.
        parts = re.split(r"(?<=[\.\!\?\؟\n])\s+", text)
        chunks: list[str] = []
        buf = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(buf) + len(part) + 1 <= limit:
                buf = (buf + " " + part).strip()
                continue
            if buf:
                chunks.append(buf)
                buf = ""
            if len(part) <= limit:
                buf = part
            else:
                # Sentence itself too long: wrap on clause/word boundaries.
                chunks.extend(_wrap_to_limit(part, limit))
        if buf:
            chunks.append(buf)
        return chunks

    @staticmethod
    def _validate_lang(language: str) -> str:
        language = (language or "").strip().lower()
        # Egyptian Arabic maps onto the base model's "ar".
        if language in ("arz", "egy", "ar-eg", "egyptian"):
            language = "ar"
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. Use one of {sorted(SUPPORTED_LANGUAGES)}."
            )
        return language

    # -- synthesis ----------------------------------------------------------
    def _prepare(self, text: str, lang: str, diacritize: bool) -> str:
        """Full text prep pipeline (order matters): sanitize -> pronunciation
        respellings -> optional Arabic diacritization. Runs before splitting so
        added diacritics are counted against the chunk length limit."""
        text = normalize_text(text)
        text = apply_pronunciation(text, lang)
        if diacritize and lang == "ar":
            text = diacritize_arabic(text)
        return text

    def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = 1.0,
        diacritize: bool = False,
    ) -> np.ndarray:
        """Generate the full utterance and return a float32 waveform (-1..1)."""
        language = self._validate_lang(language)
        voice_obj = self._resolve_voice(voice)
        chunks = self._split_text(self._prepare(text, language, diacritize), language)
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            gpt_cond_latent, speaker_embedding = self._get_latents(voice_obj.key)
            pieces: list[np.ndarray] = []
            with torch.no_grad():
                for chunk in chunks:
                    out = self.model.inference(
                        text=chunk,
                        language=language,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=temperature,
                        speed=speed,
                        **GPT_INFERENCE_PARAMS,
                    )
                    wav = out["wav"]
                    if torch.is_tensor(wav):
                        wav = wav.detach().cpu().numpy()
                    pieces.append(np.asarray(wav, dtype=np.float32).reshape(-1))
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        wav = _polish_output(np.concatenate(pieces))
        return _apply_pitch_shift(wav, voice_obj.pitch_shift)

    def stream(
        self,
        text: str,
        language: str = "en",
        voice: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = 1.0,
        diacritize: bool = False,
    ) -> Iterator[np.ndarray]:
        """Yield float32 PCM chunks as they are generated (low latency)."""
        language = self._validate_lang(language)
        voice_obj = self._resolve_voice(voice)
        chunks = self._split_text(self._prepare(text, language, diacritize), language)
        if not chunks:
            return

        has_stream = hasattr(self.model, "inference_stream")

        # IMPORTANT: we never `yield` while holding self._lock. Yielding suspends
        # the generator; if the HTTP client disconnects there, the generator may
        # never be resumed/closed and the lock would leak, deadlocking ALL future
        # TTS requests. So we generate a whole text-chunk under the lock (a bounded
        # operation that always completes and releases), then yield the buffered
        # audio afterwards — where a disconnect is harmless.
        gpt_cond_latent = None
        speaker_embedding = None
        for chunk in chunks:
            pieces: list[np.ndarray] = []
            with self._lock:
                if gpt_cond_latent is None:
                    gpt_cond_latent, speaker_embedding = self._get_latents(
                        voice_obj.key
                    )
                with torch.no_grad():
                    if has_stream:
                        for piece in self.model.inference_stream(
                            text=chunk,
                            language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=temperature,
                            speed=speed,
                            stream_chunk_size=20,
                            **GPT_INFERENCE_PARAMS,
                        ):
                            if torch.is_tensor(piece):
                                piece = piece.detach().cpu().numpy()
                            pieces.append(
                                np.asarray(piece, dtype=np.float32).reshape(-1)
                            )
                    else:
                        out = self.model.inference(
                            text=chunk,
                            language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            temperature=temperature,
                            speed=speed,
                            **GPT_INFERENCE_PARAMS,
                        )
                        wav = out["wav"]
                        if torch.is_tensor(wav):
                            wav = wav.detach().cpu().numpy()
                        pieces.append(np.asarray(wav, dtype=np.float32).reshape(-1))
            # Lock released — safe to yield to the (possibly slow/gone) client.
            # Assemble the whole sentence before yielding: pitch-shift and tail
            # cleanup are clean over sentence-length audio but click/artifact if
            # run on the tiny per-token stream pieces independently. This also
            # removes the trailing low-frequency "second voice" blip per sentence.
            if pieces:
                sentence = _polish_output(np.concatenate(pieces))
                yield _apply_pitch_shift(sentence, voice_obj.pitch_shift)


def float_to_pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype("<i2").tobytes()


def wav_bytes(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a float32 waveform as a 16-bit PCM WAV file in memory."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float_to_pcm16_bytes(wav))
    return buf.getvalue()
