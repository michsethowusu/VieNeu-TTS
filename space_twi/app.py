import os
import gradio as gr
from vieneu import Vieneu

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO = "michsethowusu/VieNeu-TTS-Twi"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
GGUF_FILES = {
    "GGUF Q4_K_M (smaller, faster)": "VieNeu-TTS-Twi-Q4_K_M.gguf",
    "GGUF Q8_0 (larger, better quality)": "VieNeu-TTS-Twi-Q8_0.gguf",
}
HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# Lazy-loaded TTS engines (one per GGUF variant)
# ---------------------------------------------------------------------------
_engines = {}

def get_engine(gguf_file: str) -> Vieneu:
    if gguf_file not in _engines:
        print(f"Loading VieNeu-TTS (standard/GGUF) with {gguf_file} ...")
        _engines[gguf_file] = Vieneu(
            mode="standard",
            backbone_repo=MODEL_REPO,
            gguf_filename=gguf_file,
            backbone_device="cpu",
            codec_repo=CODEC_REPO,
            codec_device="cpu",
            lang="twi",
            emotion="none",
            hf_token=HF_TOKEN,
        )
        print(f"Loaded {gguf_file}")
    return _engines[gguf_file]

# Eagerly load the default engine at startup so the first request is fast
# and so voice presets are available to populate the dropdown.
_default_gguf = GGUF_FILES["GGUF Q4_K_M (smaller, faster)"]
_default_engine = get_engine(_default_gguf)
VOICE_CHOICES = [vid for _desc, vid in _default_engine.list_preset_voices()]
DEFAULT_VOICE = _default_engine._default_voice or (VOICE_CHOICES[0] if VOICE_CHOICES else None)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate(text: str, voice_name: str, gguf_choice: str, temperature: float):
    if not text.strip():
        return None, "Please enter some text."

    gguf_file = GGUF_FILES[gguf_choice]
    tts = get_engine(gguf_file)

    voice = tts.get_preset_voice(voice_name)
    wav = tts.infer(
        text,
        voice=voice,
        temperature=temperature,
        apply_watermark=False,
    )
    return (tts.sample_rate, wav), f"Generated {len(wav)/tts.sample_rate:.1f}s of audio."

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="VieNeu-TTS Twi") as demo:
    gr.Markdown(
        "# 🗣️ VieNeu-TTS — Asante Twi\n"
        "Text-to-speech for Asante Twi, fine-tuned from VieNeu-TTS-0.3B. Runs on CPU via GGUF."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text (Twi)",
                placeholder="Meda wo ase paa. Onyame nhyira wo.",
                lines=3,
            )
            voice_select = gr.Dropdown(
                choices=VOICE_CHOICES,
                value=DEFAULT_VOICE,
                label="Voice",
            )
            gguf_select = gr.Dropdown(
                choices=list(GGUF_FILES.keys()),
                value="GGUF Q4_K_M (smaller, faster)",
                label="Model",
            )
            temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
            generate_btn = gr.Button("🎙️ Generate", variant="primary")

        with gr.Column():
            audio_output  = gr.Audio(label="Output", type="numpy")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[text_input, voice_select, gguf_select, temperature],
        outputs=[audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
