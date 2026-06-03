import os
import io
import numpy as np
import soundfile as sf
import gradio as gr
import modal

# ---------------------------------------------------------------------------
# Connect to the deployed Modal inference app (CPU, scale-to-zero).
# Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET as Space secrets.
# ---------------------------------------------------------------------------
MODAL_APP   = "akiti-tts"
MODAL_CLASS = "Akiti"

_akiti = None

def get_akiti():
    global _akiti
    if _akiti is None:
        Cls = modal.Cls.from_name(MODAL_APP, MODAL_CLASS)
        _akiti = Cls()
    return _akiti

# Fetch the voice list from Modal (falls back to a static list if unavailable)
try:
    VOICES = get_akiti().list_voices.remote()
except Exception as e:
    print(f"Could not fetch voices from Modal: {e}")
    VOICES = ["kofi", "abena", "akua", "kwame", "yaa"]

VOICE_CHOICES = ["No voice (default speaker)"] + VOICES

MODEL_CHOICES = {
    "Fast (Q4)": "q4",
    "Best quality (Q8)": "q8",
}

EXAMPLES = [
    "Meda wo ase paa.",
    "Akwaaba! Wo ho te sɛn?",
    "Onyame nhyira wo na wo fie.",
    "Ɛnnɛ adeɛ yɛ fɛ paa, na owia hann yɛ fɛ.",
]

# ---------------------------------------------------------------------------
# Inference (calls Modal on demand)
# ---------------------------------------------------------------------------
def synthesize(text, voice_label, model_label, temperature, top_p, rep_penalty):
    if not text or not text.strip():
        return None, "✋ Please enter some Twi text."

    voice = "none" if voice_label.startswith("No voice") else voice_label
    model = MODEL_CHOICES.get(model_label, "q4")

    try:
        wav_bytes = get_akiti().infer.remote(
            text=text, voice=voice, model=model,
            temperature=float(temperature), top_p=float(top_p), rep_pen=float(rep_penalty),
        )
    except Exception as e:
        return None, f"⚠️ Inference failed: {e}"

    wav, sr = sf.read(io.BytesIO(wav_bytes))
    return (sr, wav.astype(np.float32)), f"✅ Done — {len(wav)/sr:.1f}s of audio."

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
.gradio-container {max-width: 900px !important; margin: auto !important;}
#title {text-align:center;}
footer {visibility: hidden;}
"""

with gr.Blocks(title="Akiti TTS", theme=gr.themes.Soft(primary_hue="green"), css=CSS) as demo:
    gr.Markdown(
        "<div id='title'>\n\n"
        "# 🪘 Akiti TTS\n"
        "### Asante Twi Text-to-Speech\n"
        "Type Twi text, pick a voice, and generate natural speech. "
        "Runs on-demand — the first request after a quiet spell may take ~20–30s to warm up, "
        "then it's quick.\n\n</div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Twi text",
                placeholder="Meda wo ase paa. Onyame nhyira wo.",
                lines=4,
            )
            with gr.Row():
                voice_select = gr.Dropdown(choices=VOICE_CHOICES, value=VOICE_CHOICES[0], label="🎙️ Voice")
                model_select = gr.Dropdown(choices=list(MODEL_CHOICES.keys()),
                                           value="Fast (Q4)", label="⚙️ Model")
            with gr.Accordion("Advanced settings", open=False):
                temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature (higher = more varied)")
                top_p       = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Repetition penalty (reduces silences)")
            generate_btn = gr.Button("🎧 Generate Speech", variant="primary", size="lg")

        with gr.Column(scale=2):
            audio_output  = gr.Audio(label="Generated speech", type="numpy", autoplay=True)
            status_output = gr.Textbox(label="Status", interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

    gr.Markdown(
        "---\n"
        "Akiti TTS is fine-tuned from **VieNeu-TTS-0.3B** for Asante Twi. "
        "Model & voices: [michsethowusu/Akiti-TTS](https://huggingface.co/michsethowusu/Akiti-TTS)."
    )

    generate_btn.click(
        fn=synthesize,
        inputs=[text_input, voice_select, model_select, temperature, top_p, rep_penalty],
        outputs=[audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
