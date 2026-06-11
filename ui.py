"""
ui.py — Gradio UI for the Autonomous Video Dubbing Pipeline.

Tab 1 – Audio Generation : Text → MOSS-TTS-Nano → .wav
Tab 2 – Video Dubbing    : Video + Audio (auto-filled from Tab 1) → dubbed video
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import requests

# Single persistent output dir for TTS — old files deleted on each new generation
TTS_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="musetalk_tts_"))
_last_tts_file: Path | None = None


# ---------------------------------------------------------------------------
# Tab 1 helpers
# ---------------------------------------------------------------------------

def run_tts(text: str):
    global _last_tts_file

    if not text or not text.strip():
        raise gr.Error("Please enter some text to synthesise.")

    # Delete previous generated file
    if _last_tts_file and _last_tts_file.exists():
        _last_tts_file.unlink()

    from local_tts import generate_speech

    output_path = TTS_OUTPUT_DIR / "output.wav"
    generate_speech(
        text=text.strip(),
        output_path=str(output_path),
        mode="continuation",
        prompt_audio_path=None,
    )
    _last_tts_file = output_path
    # Return same path to both: Tab 1 audio player AND Tab 2 audio input
    return str(output_path), str(output_path)


# ---------------------------------------------------------------------------
# Tab 2 helpers
# ---------------------------------------------------------------------------

def run_dubbing(video_path, audio_path, api_url: str):
    if not video_path:
        raise gr.Error("Please upload a target video.")
    if not audio_path:
        raise gr.Error("Generate audio in Tab 1 first, or upload a WAV file.")
    if not api_url or not api_url.strip():
        raise gr.Error("Please enter the Kaggle API URL.")

    api_url = api_url.strip().rstrip("/")
    endpoint = f"{api_url}/generate"

    with open(video_path, "rb") as vf, open(audio_path, "rb") as af:
        files = {
            "video": (Path(video_path).name, vf, "video/mp4"),
            "audio": (Path(audio_path).name, af, "audio/wav"),
        }
        try:
            resp = requests.post(endpoint, files=files, timeout=300)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise gr.Error(f"Could not connect to {endpoint}. Is the Kaggle backend running?")
        except requests.exceptions.HTTPError as exc:
            raise gr.Error(f"Backend error: {exc}\n\n{resp.text[:500]}")

    output_path = TTS_OUTPUT_DIR / "dubbed_output.mp4"
    output_path.write_bytes(resp.content)
    return str(output_path)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Autonomous Video Dubbing Pipeline") as app:
        gr.Markdown(
            """
            # Autonomous Video Dubbing Pipeline
            **Phase 1** — Generate speech locally with MOSS-TTS-Nano (CPU).
            **Phase 2** — Dub a video on Kaggle T4 GPU via MuseTalk.
            """
        )

        # ── Tab 1: Audio Generation ────────────────────────────────────────
        with gr.Tab("Audio Generation"):
            gr.Markdown("### Step 1 — Generate your dubbed audio")

            with gr.Row():
                with gr.Column(scale=2):
                    tts_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter the text you want to synthesise…",
                        lines=5,
                    )

                    with gr.Group():
                        gr.Markdown(
                            "**Voice Clone Mode** — `Coming Soon` 🔒  \n"
                            "Upload a reference audio to clone a specific speaker's voice."
                        )
                        tts_prompt_audio = gr.Audio(
                            label="Reference Audio (Coming Soon)",
                            type="filepath",
                            interactive=False,
                        )

                    tts_btn = gr.Button("Generate Audio", variant="primary")

                with gr.Column(scale=1):
                    tts_output = gr.Audio(
                        label="Generated Audio — will auto-fill Tab 2",
                        type="filepath",
                    )

        # ── Tab 2: Video Dubbing ───────────────────────────────────────────
        with gr.Tab("Video Dubbing"):
            gr.Markdown(
                "### Step 2 — Dub your video\n"
                "The audio field is **auto-filled** from Tab 1. "
                "Upload a target video, paste the Kaggle ngrok URL, and click **Start Dubbing**."
            )

            with gr.Row():
                with gr.Column():
                    dub_video = gr.Video(label="Target Video", sources=["upload"])
                    # This is the explicit path wired from Tab 1's output
                    dub_audio = gr.Audio(
                        label="Dubbed Audio (auto-filled from Tab 1)",
                        type="filepath",
                        sources=["upload"],
                        interactive=True,
                    )
                    dub_api_url = gr.Textbox(
                        label="Kaggle API URL",
                        placeholder="https://xxxx-xxxx.ngrok-free.app",
                    )
                    dub_btn = gr.Button("Start Dubbing", variant="primary")

                with gr.Column():
                    dub_output = gr.Video(label="Dubbed Video Output")

        # ── Wire Tab 1 output → Tab 2 audio input ─────────────────────────
        tts_btn.click(
            fn=run_tts,
            inputs=[tts_text],
            outputs=[tts_output, dub_audio],   # fills both simultaneously
        )

        dub_btn.click(
            fn=run_dubbing,
            inputs=[dub_video, dub_audio, dub_api_url],
            outputs=dub_output,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False, theme=gr.themes.Soft())
