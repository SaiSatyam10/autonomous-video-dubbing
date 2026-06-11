# Autonomous Video Dubbing Pipeline — Architecture

> **Legend:** ✅ Done · ⬜ Pending · 🔄 In Progress

---

## System Overview

```
┌─────────────────────────────────┐        ┌──────────────────────────────────────┐
│         LOCAL MACHINE           │        │          KAGGLE (1x T4 GPU)          │
│                                 │        │                                      │
│  ui.py (Gradio)                 │        │  app.py (FastAPI)                    │
│  ┌─────────────┐                │        │  ┌──────────────────────────────┐    │
│  │ Tab 1: TTS  │ local_tts.py   │        │  │  POST /generate              │    │
│  │ MOSS-TTS-   │────────────►   │        │  │  sanitizer.py                │    │
│  │ Nano (CPU)  │   .wav file    │        │  │  MuseTalk inference           │    │
│  └─────────────┘                │  HTTP  │  │  (cuda:0, FP16)              │    │
│  ┌───────────────┐              │ POST ► │  └──────────────────────────────┘    │
│  │ Tab 2: Dub    │ video+audio  │        │           │                          │
│  │ Video Upload  │──────────────┼────────┼──►        │ pyngrok tunnel           │
│  │ Kaggle URL    │◄─────────────┼────────┼──  dubbed video (.mp4)               │
│  └───────────────┘   response  │        │                                      │
└─────────────────────────────────┘        └──────────────────────────────────────┘
                                                        │
                                           ┌────────────▼────────────┐
                                           │   GitHub Pages (Phase 3) │
                                           │   index.html + result MP4s │
                                           └─────────────────────────┘
```

---

## Phase 1 — Local Environment & Audio Generation ✅

### 1.1 Project Initialisation ✅
- ✅ Created `venv` with Python 3.11
- ✅ Created `requirements.txt` (fastapi, uvicorn, gradio, requests)
- ✅ Installed dependencies including torch 2.7.0, torchaudio, transformers 4.57.1

### 1.2 MOSS-TTS-Nano Integration ✅
- ✅ Cloned `https://github.com/OpenMOSS/MOSS-TTS-Nano.git`
- ✅ Model: `OpenMOSS-Team/MOSS-TTS-Nano` (0.1B params, CPU-capable)
- ✅ Audio tokenizer: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (~20M params)
- ✅ `WeTextProcessing` skipped on macOS (requires `pynini` native lib); replaced with built-in robust text normaliser
- ✅ Wrote `local_tts.py`:
  - Lazy model singleton (loaded once on first call)
  - `generate_speech(text, output_path, mode, prompt_audio_path)` public API
  - Supports `continuation` (plain TTS) and `voice_clone` modes
  - Device auto-detection (CUDA > CPU), dtype auto-selection
  - CLI interface: `python local_tts.py --text "..." --output out.wav`
- ✅ Smoke-tested: generated `tts_test.wav` successfully on CPU

### 1.3 Local Gradio UI (`ui.py`) ✅
- ✅ Tab 1 — Audio Generation:
  - Textbox input
  - Mode selector (continuation / voice_clone)
  - Reference audio upload (visible only in voice_clone mode)
  - "Generate Audio" button → calls `local_tts.generate_speech`
  - Audio output component
- ✅ Tab 2 — Video Dubbing:
  - Video upload (target)
  - Audio upload (generated from Tab 1 or custom)
  - Kaggle API URL text field (ngrok URL)
  - "Start Dubbing" → POSTs `multipart/form-data` to `{url}/generate`
  - Video output component for dubbed result
- ✅ Gradio 6.13.0, soft theme, responsive layout

---

## Phase 2 — Kaggle Backend (MuseTalk on T4) ✅

### 2.1 Environment Setup ✅
- ✅ `kaggle_musetalk.ipynb` — single notebook containing all Phase 2 logic
- ✅ Force single GPU: `CUDA_VISIBLE_DEVICES=0` at kernel start (Cell 2)
- ✅ Clone `https://github.com/TMElyralab/MuseTalk.git` (Cell 3)
- ✅ Dependency install cell (Cell 4):
  - `pip install -r MuseTalk/requirements.txt`
  - `pip install fastapi uvicorn python-multipart pyngrok`

### 2.2 Model Weight Loader (`model_loader.py` logic) ✅
- ✅ Check `/kaggle/input/` for any subdirectory containing a `models/musetalkV15/` folder (Cell 5)
- ✅ **IF found**: symlink `/kaggle/input/<dataset>/models/` → `MuseTalk/models/`; falls back to `shutil.copytree` on cross-device error
- ✅ **ELSE**: run `sh MuseTalk/download_weights.sh` from the current upstream repo to fetch the recommended MuseTalk 1.5 asset bundle from Hugging Face
- ✅ Preferred lip-sync checkpoint for this multilingual pipeline: `models/musetalkV15/unet.pth`
- ✅ Cell 6: Model verification should assert the MuseTalk 1.5 checkpoint and its companion assets exist and are non-zero bytes
- ✅ DWPose note: this checked-out MuseTalk codebase currently loads the PyTorch checkpoint `models/dwpose/dw-ll_ucoco_384.pth`; do not expect the ONNX pair here unless you also switch the inference stack

  Expected model structure after loading:
  ```
  MuseTalk/models/
    musetalkV15/
      musetalk.json
      unet.pth
    syncnet/
      latentsync_syncnet.pt
    dwpose/
      dw-ll_ucoco_384.pth
    face-parse-bisent/
      79999_iter.pth
      resnet18-5c106cde.pth
    sd-vae/
      config.json
      diffusion_pytorch_model.bin
    whisper/
      config.json
      preprocessor_config.json
      pytorch_model.bin
  ```

  Download source:
  ```
  git clone https://github.com/TMElyralab/MuseTalk.git
  cd MuseTalk
  sh download_weights.sh
  ```

### 2.3 Pre-processing Sanitiser (`sanitizer.py` logic) ✅
- ✅ `sanitize_video()`: ffmpeg resize to 512×512 with letterboxing; strips audio track
- ✅ `sanitize_audio()`: ffmpeg → 16kHz mono PCM WAV; ffprobe duration check (min 0.5s)
- ✅ Written to `/kaggle/working/sanitizer.py` via Cell 7 (no OpenCV needed — pure ffmpeg)

### 2.4 FastAPI Application (`app.py`) ✅
- ✅ Written to `/kaggle/working/app.py` via Cell 8
- ✅ `CUDA_VISIBLE_DEVICES=0` enforced in-process and at kernel level
- ✅ MuseTalk called via `subprocess.run(["python", "-m", "scripts.inference", ...], cwd=MuseTalkDir)` — preserves relative imports; target MuseTalk 1.5 (`models/musetalkV15`, `--version v15`)
- ✅ `POST /generate`: upload video + audio → sanitise → inference → `FileResponse(mp4)`
- ✅ Output search handles nested result dirs (MuseTalk version variations)
- ✅ Temp work dir cleaned after each request; permanent copy in `RESULT_DIR`
- ✅ `GET /health`: returns GPU name + free VRAM
- ✅ pyngrok tunnel (Cell 9): exposes port 8000, prints public URL + health/generate URLs

### 2.5 Hardware Targets ✅
- ✅ `CUDA_VISIBLE_DEVICES=0` (only GPU 0 of the 2×T4 instance)
- ✅ Configurable `BATCH_SIZE` (default 8, reduce to 4 if OOM)
- ✅ Cell 10: heartbeat loop prints VRAM usage every 60s for monitoring
- ✅ Target: 16 GB VRAM budget (FP16 flag passed to inference subprocess)

---

## Phase 3 — Static Results Site (GitHub Pages) ✅

### 3.1 Frontend Scaffold ✅
- ✅ `index.html` — side-by-side source/output comparison page
- ✅ Responsive static styling
- ✅ Sections for system flow and generated results

### 3.2 Generated Results ✅
- ✅ Source video, driving audio, and generated output assets
- ✅ Examples configured through `Githubio_code/examples.js`

### 3.3 Deployment ✅
- ✅ Static result files published through GitHub Pages
- ✅ Live URL: `https://saisatyamjena.github.io/GeneratedLipSyncSamples/`

---

## Data Flow — Request Lifecycle

```
User types text
    │
    ▼
[Tab 1] local_tts.generate_speech()
    │  MOSS-TTS-Nano (CPU, ~2-5s)
    ▼
generated.wav (48kHz stereo)
    │
    ├──► User downloads / listens
    │
    ▼ (Tab 2)
User uploads target.mp4 + generated.wav
    │
    ▼
POST /generate  ──────────────────────────────────────►  Kaggle FastAPI
                                                              │
                                                         sanitizer.py
                                                         ├─ resize video → 512×512
                                                         ├─ audio → 16kHz mono WAV
                                                         └─ validate inputs
                                                              │
                                                         MuseTalk inference
                                                         ├─ Whisper: audio → phoneme timestamps
                                                         ├─ DWPose: face landmark detection
                                                         ├─ VAE: encode/decode lip region
                                                         └─ UNet: generate lip frames (FP16, cuda:0)
                                                              │
                                                         ffmpeg: merge audio + frames → .mp4
                                                              │
◄──────────────────────────────────────────────────  FileResponse(dubbed.mp4)
    │
[Tab 2] gr.Video output
```

---

## File Map

```
project-root/
├── venv/                          ✅  Python 3.11 virtual environment
├── requirements.txt               ✅  Local deps (gradio, fastapi, requests, uvicorn)
├── local_tts.py                   ✅  MOSS-TTS-Nano wrapper (generate_speech API + CLI)
├── ui.py                          ✅  Gradio UI (Tab1: TTS, Tab2: Dubbing)
├── architecture.md                ✅  This file
├── MOSS-TTS-Nano/                 ✅  Tracked TTS source snapshot
│   ├── moss_tts_nano/             ✅  Core model package
│   ├── infer.py                   ✅  Reference inference script
│   ├── text_normalization_pipeline.py  ✅
│   └── assets/audio/              ✅  Sample reference audio files
│
├── kaggle_musetalk.ipynb          ✅  Kaggle Phase 2 notebook (10 cells)
├── MuseTalk/                      ✅  Tracked inference source and configs
│   ├── scripts/inference.py       ✅  Main generation entrypoint
│   ├── musetalk/models/           ✅  VAE and UNet implementations
│   ├── musetalk/utils/            ✅  Audio, pose, parsing and blending
│   └── LICENSE                    ✅  Upstream project license
│
├── Githubio_code/                 ✅  Static results-site source
│   ├── index.html                 ✅
│   ├── style.css                  ✅
│   └── examples.js                ✅
└── Githubio_assets/               ✅  Selected generated-result media
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| MOSS-TTS-Nano on CPU locally | 0.1B params, fast on CPU, no local GPU required |
| `WeTextProcessing` disabled on macOS | `pynini` requires native C++ libs not available on macOS without brew |
| Single T4 (`cuda:0`) on Kaggle 2xT4 | Simpler memory management; MuseTalk pipeline fits in 16GB FP16 |
| 512×512 video resize | MuseTalk's UNet OOMs above this resolution on T4 |
| 16 kHz mono WAV for audio input | Whisper (used by MuseTalk for timing) is trained on 16kHz |
| pyngrok tunnel | Exposes Kaggle's sandboxed port 8000 without SSH or static IP |
| Gradio 6.13 for UI | No custom JS required; native audio/video components |
