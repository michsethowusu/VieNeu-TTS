"""
Gradio app for VieNeu-TTS Twi — runs on HF Space or locally.

Supports three inference modes:
  • CPU  — PyTorch float32, no GPU needed (slow, ~1-2 min/utterance)
  • CPU Fast (GGUF)  — Q4_K_M quantized via llama-cpp, ~10-30s on CPU
  • GPU  — PyTorch bfloat16 on CUDA (fast, requires GPU Space)

The app reads voices.json from the model repo on HF Hub and lets the user
pick a voice preset and enter Twi text.
"""

import os
import json
import tempfile
import logging
import numpy as np
import gradio as gr
import soundfile as sf
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gradio_twi")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_REPO  = os.getenv("MODEL_REPO", "michsethowusu/VieNeu-TTS-Twi")
GGUF_FILE   = os.getenv("GGUF_FILE",  "VieNeu-TTS-Twi-Q4_K_M.gguf")
CODEC_REPO  = "neuphonic/neucodec-onnx-decoder-int8"   # lightweight ONNX, CPU-friendly

HAS_GPU = False
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    pass

MODES = {}
MODES["CPU (Standard)"] = dict(
    backbone_device="cpu",
    gguf_filename=None,
    codec_device="cpu",
    description="PyTorch CPU — works everywhere, ~1-2 min per sentence",
)
try:
    import llama_cpp  # noqa: F401
    MODES["CPU Fast (GGUF Q4)"] = dict(
        backbone_device="cpu",
        gguf_filename=GGUF_FILE,
        codec_device="cpu",
        description="Quantized GGUF — ~10-30s per sentence on CPU",
    )
except ImportError:
    pass

if HAS_GPU:
    MODES["GPU"] = dict(
        backbone_device="cuda",
        gguf_filename=None,
        codec_device="cuda",
        description="PyTorch CUDA — fastest, requires GPU",
    )

DEFAULT_MODE = "CPU Fast (GGUF Q4)" if "CPU Fast (GGUF Q4)" in MODES else "CPU (Standard)"

# ---------------------------------------------------------------------------
# Load voices from HF Hub
# ---------------------------------------------------------------------------

def _load_voices():
    try:
        path = hf_hub_download(MODEL_REPO, "voices.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Could not load voices.json from {MODEL_REPO}: {e}")
        return {"default_voice": None, "presets": {}}

_voices_data = _load_voices()
VOICE_NAMES  = list(_voices_data.get("presets", {}).keys())
DEFAULT_VOICE = _voices_data.get("default_voice") or (VOICE_NAMES[0] if VOICE_NAMES else None)

# ---------------------------------------------------------------------------
# Lazy model cache — one instance per mode key
# ---------------------------------------------------------------------------

_model_cache: dict = {}

def _get_model(mode_key: str):
    if mode_key not in _model_cache:
        from vieneu import Vieneu
        cfg = MODES[mode_key]
        logger.info(f"Loading model for mode '{mode_key}' ...")
        _model_cache[mode_key] = Vieneu(
            mode="standard",
            backbone_repo=MODEL_REPO,
            backbone_device=cfg["backbone_device"],
            codec_repo=CODEC_REPO,
            codec_device=cfg["codec_device"],
            gguf_filename=cfg.get("gguf_filename"),
            lang="twi",
            emotion=None,
        )
        logger.info(f"Model loaded for mode '{mode_key}'")
    return _model_cache[mode_key]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(text: str, voice_name: str, mode_key: str):
    if not text.strip():
        return None, "Please enter some text."
    if not VOICE_NAMES:
        return None, "No voices available. Check that voices.json was pushed to HF."
    if mode_key not in MODES:
        return None, f"Mode '{mode_key}' not available in this environment."

    try:
        tts   = _get_model(mode_key)
        voice = _voices_data["presets"][voice_name]
        audio = tts.infer(text, voice=voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 24000)
            return f.name, f"✅ Generated {len(audio)/24000:.1f}s of audio"
    except Exception as e:
        logger.exception("Inference error")
        return None, f"❌ Error: {e}"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_TEXTS = [
    "Nanso Petro san hyɛɛ Kristofo nkuran sɛ monni nnipa nyinaa ni.",
    "Yesu a ɔyɛ yɛn hene no ayɛ nhwɛsoɔ ama yɛn.",
    "Meda wo ase paa.",
]

with gr.Blocks(title="VieNeu-TTS Twi") as demo:
    gr.Markdown(
        """
        # 🗣️ VieNeu-TTS — Twi
        Text-to-speech for Asante Twi, fine-tuned from [VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B).
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Twi text",
                placeholder="Enter Twi text here...",
                lines=3,
            )
            with gr.Row():
                voice_select = gr.Dropdown(
                    choices=VOICE_NAMES,
                    value=DEFAULT_VOICE,
                    label="Voice",
                )
                mode_select = gr.Dropdown(
                    choices=list(MODES.keys()),
                    value=DEFAULT_MODE,
                    label="Inference mode",
                )
            mode_info = gr.Markdown(
                f"_{MODES[DEFAULT_MODE]['description']}_"
            )
            generate_btn = gr.Button("Generate speech", variant="primary")

        with gr.Column(scale=2):
            audio_output = gr.Audio(label="Output audio", type="filepath")
            status_box   = gr.Textbox(label="Status", interactive=False)

    gr.Examples(
        examples=[[t, DEFAULT_VOICE, DEFAULT_MODE] for t in EXAMPLE_TEXTS],
        inputs=[text_input, voice_select, mode_select],
        label="Example texts",
    )

    def _update_mode_info(mode_key):
        return f"_{MODES.get(mode_key, {}).get('description', '')}_"

    mode_select.change(_update_mode_info, inputs=mode_select, outputs=mode_info)

    generate_btn.click(
        generate,
        inputs=[text_input, voice_select, mode_select],
        outputs=[audio_output, status_box],
    )

if __name__ == "__main__":
    demo.launch()
