#!/usr/bin/env python
"""
Verification script for the NileTTS-XTTS environment.

Loads the fine-tuned Coqui XTTS v2 model (KickItLikeShika/NileTTS-XTTS),
computes conditioning latents from a reference voice, then synthesizes one
English and one Egyptian Arabic utterance on the GPU, saving each to a WAV and
reporting wall-clock time, audio duration, and real-time factor (RTF).

Run:  ./venv/bin/python scripts/verify_tts.py
"""

from __future__ import annotations

import os
import time

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "nile-xtts")
VOICE_WAV = os.path.join(BASE_DIR, "voices", "default.wav")
OUT_EN = os.path.join(BASE_DIR, "scripts", "out_en.wav")
OUT_AR = os.path.join(BASE_DIR, "scripts", "out_ar.wav")

SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# PyTorch 2.6+ safe-unpickling fix.
#
# torch>=2.6 flipped torch.load's default to weights_only=True, which refuses
# to unpickle the XTTS checkpoint / config objects. We register the XTTS config
# classes as safe globals AND force weights_only=False on torch.load as a
# belt-and-suspenders fallback. This must run before load_checkpoint().
# ---------------------------------------------------------------------------
def install_torch_load_fix() -> None:
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig

        torch.serialization.add_safe_globals(
            [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
        )
    except Exception as e:  # pragma: no cover
        print("[warn] add_safe_globals partial:", e)

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load


def main() -> None:
    print("torch", torch.__version__, "| torchaudio", torchaudio.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    install_torch_load_fix()

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    # --- load model --------------------------------------------------------
    t0 = time.time()
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print(f"[load] model ready in {time.time() - t0:.1f}s")

    # --- conditioning latents from reference voice -------------------------
    import soundfile as sf

    info = sf.info(VOICE_WAV)
    print(f"[voice] {VOICE_WAV}: {info.samplerate} Hz, "
          f"{info.frames / info.samplerate:.2f}s, {info.channels}ch")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=VOICE_WAV, gpt_cond_len=6, max_ref_length=30
    )
    print("[latents] gpt_cond_latent", tuple(gpt_cond_latent.shape),
          "| speaker_embedding", tuple(speaker_embedding.shape))

    # --- inference availability check --------------------------------------
    print("[stream] inference_stream present:",
          hasattr(model, "inference_stream"))

    cases = [
        ("en", "Hello, this is a test of the Nile text to speech system.", OUT_EN),
        ("ar", "مرحبا، إزيك النهارده؟ ده اختبار للنظام.", OUT_AR),
    ]

    for lang, text, out_path in cases:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        with torch.no_grad():
            out = model.inference(
                text=text,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gen = time.time() - t0

        wav = out["wav"]
        if torch.is_tensor(wav):
            wav = wav.detach().cpu()
        else:
            import numpy as np
            wav = torch.from_numpy(np.asarray(wav, dtype="float32"))
        wav = wav.reshape(1, -1)  # (channels, samples) for torchaudio.save

        torchaudio.save(out_path, wav, SAMPLE_RATE)
        dur = wav.shape[-1] / SAMPLE_RATE
        size = os.path.getsize(out_path)
        rtf = gen / dur if dur else float("nan")
        print(f"[{lang}] type(out['wav'])={type(out['wav']).__name__} "
              f"shape={tuple(wav.shape)} | wrote {out_path} "
              f"({size} bytes) | audio {dur:.2f}s | gen {gen:.2f}s | RTF {rtf:.3f}")

    print("DONE")


if __name__ == "__main__":
    main()
