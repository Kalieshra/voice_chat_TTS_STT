# NileTTS — English / Arabic / Egyptian Text-to-Speech

A fast, GPU-accelerated Text-to-Speech web app built on
[**NileTTS-XTTS**](https://huggingface.co/KickItLikeShika/NileTTS-XTTS) — a fine-tune of
Coqui **XTTS v2** specialized for **Egyptian Arabic**, which also speaks **English** and
**Modern Standard Arabic**. Type text, pick a language, hit **Speak**, and hear it stream
back within a fraction of a second. Tuned for an **RTX 4090**.

## Features
- 🌍 **English (`en`)** and **Arabic / Egyptian dialect (`ar`)** in one model.
- ⚡ **Streaming synthesis** — audio starts playing while the rest is still generating.
- 🚀 **RTX 4090 tuning** — TF32 matmuls, model kept warm in VRAM, and **conditioning
  latents cached per voice** so each request is just a forward pass.
- 🗣️ **Voice cloning** — bundled default voice, plus optional 6-second reference upload.
- 🎚️ Adjustable speed; RTL-aware Arabic text box; live latency/RTF metrics.

## Architecture
```
app/tts_engine.py   NileTTSEngine — loads XTTS once, caches latents, full + streaming synth
app/server.py       FastAPI: /api/tts, /api/tts/stream, /api/voices, /api/upload-voice
app/static/         Single-page web UI (Web Audio streaming player)
models/nile-xtts/   config.json, model.pth, vocab.json  (downloaded from HF)
voices/default.wav  Reference speaker for zero-shot cloning
scripts/verify_tts.py   Standalone GPU inference sanity check
```

## Setup
```bash
cd /home/nova/proj/Text_To_Speach
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

# 1) PyTorch with CUDA (matches RTX 4090 / driver 595)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2) App deps
pip install -r requirements.txt

# 3) API keys for the chat/agent features (optional for pure TTS)
cp .env.example .env      # then edit .env and fill in your key(s)
```

### Model weights (~5.6 GB)
The XTTS checkpoint is **not** in this repo (too large for GitHub). It is fetched
from Hugging Face automatically **on first startup** into `models/nile-xtts/`, and
the web UI shows a live download progress bar. To pre-download it instead:

```bash
huggingface-cli download KickItLikeShika/NileTTS-XTTS --local-dir models/nile-xtts
```

Override the source repo with `NILE_MODEL_REPO=<org/repo>` if you host your own.

## Run
```bash
./run.sh
# then open http://localhost:8000  (model downloads on first run; UI shows progress)
```

## Notes
- The model needs a reference `.wav` for its zero-shot voice; `voices/default.wav` is used
  unless you upload your own. Egyptian Arabic just means writing colloquial text with
  language `ar`.
- First request after startup is warmed automatically during model load.
- Set `NILE_DEEPSPEED=1` before `run.sh` to enable DeepSpeed GPT acceleration (optional).
