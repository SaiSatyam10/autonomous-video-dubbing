"""
local_tts.py — Generate a .wav file from text using MOSS-TTS-Nano.

Usage as a library:
    from local_tts import generate_speech
    output_path = generate_speech("Hello world", output_path="out.wav")

Usage from CLI:
    python local_tts.py --text "Hello world" --output out.wav
"""

from __future__ import annotations

import sys
import argparse
import logging
import os
from pathlib import Path

# Ensure the MOSS-TTS-Nano package is importable
PROJECT_DIR = Path(__file__).resolve().parent
MOSS_TTS_DIR = PROJECT_DIR / "MOSS-TTS-Nano"
if str(MOSS_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(MOSS_TTS_DIR))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model singleton — loaded once on first call
# ---------------------------------------------------------------------------
_model = None
_device = None
_dtype = None


def _load_model():
    global _model, _device, _dtype
    if _model is not None:
        return _model, _device, _dtype

    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # CPU-safe default; bfloat16 on CUDA

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info("Loading MOSS-TTS-Nano on %s (%s)…", device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        "OpenMOSS-Team/MOSS-TTS-Nano",
        trust_remote_code=True,
    )
    model.to(device=device, dtype=dtype)
    model._set_attention_implementation("sdpa")
    model.eval()

    _model, _device, _dtype = model, device, dtype
    logger.info("MOSS-TTS-Nano loaded.")
    return model, device, dtype


def generate_speech(
    text: str,
    output_path: str | Path = "generated.wav",
    mode: str = "continuation",
    prompt_audio_path: str | Path | None = None,
) -> str:
    """
    Synthesise *text* and save the result to *output_path*.

    Parameters
    ----------
    text:
        The text to synthesise.
    output_path:
        Destination .wav file (created / overwritten).
    mode:
        "continuation" for plain TTS; "voice_clone" to clone *prompt_audio_path*.
    prompt_audio_path:
        Reference audio for voice-clone mode (ignored in continuation mode).

    Returns
    -------
    str
        Absolute path of the written .wav file.
    """
    from text_normalization_pipeline import prepare_tts_request_texts

    output_path = str(Path(output_path).resolve())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model, device, _ = _load_model()

    # Text normalisation (no WeTextProcessing — not available on macOS)
    prepared = prepare_tts_request_texts(
        text=text,
        prompt_text="",
        voice="",
        enable_wetext=False,
        enable_normalize_tts_text=True,
        text_normalizer_manager=None,
    )
    normalised_text = str(prepared["text"])

    logger.info("Synthesising %d chars in mode=%s …", len(normalised_text), mode)

    result = model.inference(
        text=normalised_text,
        output_audio_path=output_path,
        mode=mode,
        prompt_text=None,
        prompt_audio_path=str(prompt_audio_path) if prompt_audio_path else None,
        audio_tokenizer_type="moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device=device,
        max_new_frames=375,
        do_sample=True,
    )

    logger.info("Saved audio → %s", result["audio_path"])
    return str(result["audio_path"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate speech with MOSS-TTS-Nano")
    p.add_argument("--text", required=True, help="Text to synthesise")
    p.add_argument("--output", default="generated.wav", help="Output .wav path")
    p.add_argument(
        "--mode",
        default="continuation",
        choices=["continuation", "voice_clone"],
        help="Inference mode",
    )
    p.add_argument("--prompt-audio", default=None, help="Reference audio for voice_clone mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = generate_speech(
        text=args.text,
        output_path=args.output,
        mode=args.mode,
        prompt_audio_path=args.prompt_audio,
    )
    print(f"Output: {path}")
