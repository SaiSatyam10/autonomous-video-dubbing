# Autonomous Video Dubbing

An end-to-end video dubbing system that converts text into speech and
synchronizes the generated audio with a speaker's lip movements.

The project combines
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) for speech synthesis,
[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) for audio-driven lip
synchronization, Gradio for the user interface, and a FastAPI service running
on a Kaggle GPU.

**Generated results:** https://saisatyamjena.github.io/GeneratedLipSyncSamples/

## System Overview

```text
Text
  |
  v
MOSS-TTS-Nano
  |  speech waveform
  v
Gradio interface
  |  target video + generated audio
  v
FastAPI service on Kaggle T4
  |
  +-- FFmpeg input normalization
  +-- Whisper audio feature extraction
  +-- DWPose facial landmark detection
  +-- VAE face-latent encoding
  +-- Audio-conditioned MuseTalk UNet inference
  +-- VAE decoding and face-region blending
  +-- FFmpeg audio/video composition
  |
  v
Lip-synchronized video
```

The system separates lightweight local interaction from GPU-intensive video
generation. Speech synthesis and the Gradio interface run locally, while
MuseTalk inference runs on a Kaggle T4 GPU and is accessed through an HTTPS
API.

## Core Components

### Speech synthesis

[`local_tts.py`](local_tts.py) provides the project-level interface to
MOSS-TTS-Nano. It:

- loads `OpenMOSS-Team/MOSS-TTS-Nano` through Hugging Face Transformers;
- selects CPU or CUDA and a compatible tensor data type;
- keeps the model in memory between generations;
- normalizes input text before inference;
- supports continuation and reference-audio inference modes; and
- writes the generated speech to a WAV file.

The included [`MOSS-TTS-Nano/`](MOSS-TTS-Nano/) source contains the model
runtime, text-normalization pipeline, command-line interface, ONNX utilities,
and fine-tuning tools.

### Lip synchronization

[`kaggle_musetalk.ipynb`](kaggle_musetalk.ipynb) provisions the GPU inference
service and documents the complete MuseTalk execution path. It configures the
runtime, validates model checkpoints, normalizes input media, creates the
FastAPI service, and exposes the service through ngrok.

The included [`MuseTalk/`](MuseTalk/) source provides the inference pipeline and
its supporting model implementations:

| Component | Role |
| --- | --- |
| Whisper | Converts speech into time-aligned audio features |
| DWPose | Detects facial landmarks and identifies the generation region |
| Stable Diffusion VAE | Encodes face crops into latents and decodes generated latents |
| MuseTalk UNet | Predicts lip-region latents conditioned on audio features |
| Face parser | Builds masks for blending generated regions into source frames |
| FFmpeg | Normalizes media and combines generated frames with audio |

### Application interface

[`ui.py`](ui.py) implements a two-stage Gradio workflow:

1. **Audio Generation** converts entered text into speech.
2. **Video Dubbing** sends a target video and driving audio to the GPU service
   and displays the returned video.

The generated audio is passed directly from the first stage to the second, and
users can also provide their own WAV file.

### Inference API

The Kaggle notebook creates a FastAPI application with two endpoints:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Reports service availability and GPU memory |
| `POST` | `/generate` | Accepts a video and WAV file and returns a lip-synchronized MP4 |

For each generation request, the service:

1. creates an isolated working directory;
2. reads the source video's original resolution;
3. converts the video to a letterboxed `512 x 512` representation;
4. converts the audio to 16 kHz mono PCM WAV;
5. writes a MuseTalk inference configuration;
6. runs MuseTalk 1.5 in FP16 mode;
7. restores the generated video to the source resolution; and
8. returns the final MP4.

## Model Inference Flow

MuseTalk generation is performed by
[`MuseTalk/scripts/inference.py`](MuseTalk/scripts/inference.py):

1. Whisper extracts audio embeddings from the driving speech.
2. Face detection and DWPose locate the face and lip region in each frame.
3. The VAE encodes masked and reference face crops into latent tensors.
4. Positional encoding adds temporal information to the audio features.
5. The MuseTalk UNet predicts lip-region latents conditioned on those features.
6. The VAE decodes the predicted latents into image regions.
7. Face-parsing masks blend the generated regions into the original frames.
8. FFmpeg combines the processed frames with the driving audio.

The notebook includes a model-initialization map linking each neural component
to its configuration, checkpoint, implementation, and role in this sequence.

## Repository Structure

```text
.
├── ui.py
├── local_tts.py
├── requirements.txt
├── kaggle_musetalk.ipynb
├── architecture.md
├── MuseTalk/
│   ├── scripts/inference.py
│   ├── musetalk/models/
│   ├── musetalk/utils/
│   ├── configs/
│   └── LICENSE
├── MOSS-TTS-Nano/
│   ├── infer.py
│   ├── moss_tts_nano_runtime.py
│   ├── text_normalization_pipeline.py
│   ├── moss_tts_nano/
│   ├── finetuning/
│   └── LICENSE
├── Githubio_code/
└── Githubio_assets/
```

## Requirements

- Python 3.11 or 3.12
- FFmpeg and `ffprobe`
- A Kaggle notebook with a GPU accelerator
- An ngrok account for exposing the inference API
- Internet access for dependency and model retrieval

MuseTalk requires a CUDA-capable GPU. MOSS-TTS-Nano can run on CPU, although
CUDA can reduce generation time.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/SaiSatyam10/autonomous-video-dubbing.git
cd autonomous-video-dubbing

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Install the application and MOSS-TTS-Nano dependencies:

```bash
python -m pip install -r MOSS-TTS-Nano/requirements.txt
python -m pip install -r requirements.txt
```

FFmpeg must be available on the command line. For example, on macOS:

```bash
brew install ffmpeg
```

## Generate Speech

Run MOSS-TTS-Nano directly from the command line:

```bash
python local_tts.py \
  --text "Speech generation for an audio-driven video." \
  --output generated.wav
```

On first use, Hugging Face downloads the MOSS-TTS-Nano model and its audio
tokenizer. Subsequent calls reuse the local model cache.

## Run the Application

### 1. Start the MuseTalk service

Import [`kaggle_musetalk.ipynb`](kaggle_musetalk.ipynb) into Kaggle, enable a
GPU accelerator, provide the required MuseTalk model files, configure the
ngrok token, and run the notebook cells in order.

The notebook prints the public API URL after Uvicorn and ngrok start. The
notebook must remain active while inference requests are processed.

### 2. Start the Gradio interface

```bash
python ui.py
```

Open the local Gradio URL, generate or upload the driving audio, upload a target
video, enter the API base URL, and start generation.

## Model Files

Model checkpoints are distributed separately from this repository. MuseTalk
expects the following model layout:

```text
MuseTalk/models/
├── musetalkV15/
│   ├── musetalk.json
│   └── unet.pth
├── dwpose/
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent/
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
├── whisper/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── pytorch_model.bin
└── syncnet/
    └── latentsync_syncnet.pt
```

The required files can be obtained using the download instructions provided by
the upstream MuseTalk project.

## Results

Generated examples are published at:

https://saisatyamjena.github.io/GeneratedLipSyncSamples/

The site presents source videos, driving audio, generated outputs, and the
processing flow. Its static source and selected media are available in
[`Githubio_code/`](Githubio_code/) and
[`Githubio_assets/`](Githubio_assets/).

## Technologies

- Python and PyTorch
- Hugging Face Transformers
- MOSS-TTS-Nano
- MuseTalk 1.5
- Whisper
- Stable Diffusion VAE
- DWPose and MMPose
- Gradio
- FastAPI and Uvicorn
- FFmpeg
- Kaggle GPU
- ngrok

## Upstream Projects

This repository integrates source from:

- [MuseTalk](https://github.com/TMElyralab/MuseTalk), provided under its
  included license.
- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), provided under
  its included Apache 2.0 license.

Review the upstream code and model licenses before redistribution or commercial
use.

## Responsible Use

Use this system only with video, audio, and voices that you own or have
permission to process. Synthetic or modified media should be clearly
identified, and the system must not be used for impersonation, fraud,
harassment, or deceptive content.
