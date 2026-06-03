import os
import re
import json
import io
import numpy as np
import gradio as gr
import soundfile as sf
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO    = "michsethowusu/VieNeu-TTS-Twi"
GGUF_Q4       = "VieNeu-TTS-Twi-Q4_K_M.gguf"
GGUF_Q8       = "VieNeu-TTS-Twi-Q8_0.gguf"
CODEC_REPO    = "neuphonic/neucodec-onnx-decoder-int8"
VOICES_FILE   = "voices.json"
SAMPLE_RATE   = 24000
MAX_REF_CODES = 200

# ---------------------------------------------------------------------------
# Load voices.json from HF
# ---------------------------------------------------------------------------
def load_voices():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=VOICES_FILE, repo_type="model")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

VOICES = load_voices()
VOICE_NAMES = list(VOICES["presets"].keys())

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_model_cache = {}
_codec_cache = {}

def get_codec():
    if "codec" not in _codec_cache:
        from vieneu.utils import NeuCodecOnnx
        _codec_cache["codec"] = NeuCodecOnnx.from_pretrained(CODEC_REPO)
    return _codec_cache["codec"]

def get_model(gguf_file: str):
    if gguf_file not in _model_cache:
        from llama_cpp import Llama
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=gguf_file, repo_type="model")
        print(f"Loading {gguf_file} ...")
        _model_cache[gguf_file] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False,
        )
        print(f"Loaded {gguf_file}")
    return _model_cache[gguf_file]

# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------
def phonemize(text: str) -> str:
    from phonemizer import phonemize as _ph
    return _ph(text, backend="espeak", language="lfn", with_stress=True, preserve_punctuation=True)

# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
SPEECH_START = "<|SPEECH_GENERATION_START|>"
SPEECH_END   = "<|SPEECH_GENERATION_END|>"
TEXT_START   = "<|TEXT_PROMPT_START|>"
TEXT_END     = "<|TEXT_PROMPT_END|>"

def build_prompt(ref_phones: str, target_phones: str, ref_codes: list) -> str:
    codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes[:MAX_REF_CODES])
    return (
        f"{TEXT_START}{ref_phones.strip()} {target_phones.strip()}{TEXT_END}"
        f"{SPEECH_START}{codes_str}"
    )

def extract_codes(tokens: list) -> list:
    codes = []
    for tok in tokens:
        m = re.match(r"<\|speech_(\d+)\|>", tok or "")
        if m:
            codes.append(int(m.group(1)))
    return codes

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate(text: str, voice_name: str, gguf_choice: str,
             temperature: float, top_p: float, repetition_penalty: float):
    if not text.strip():
        return None, "Please enter some text."

    gguf_file = GGUF_Q4 if "Q4" in gguf_choice else GGUF_Q8
    llm   = get_model(gguf_file)
    codec = get_codec()

    voice      = VOICES["presets"][voice_name]
    ref_codes  = voice["codes"][:MAX_REF_CODES]
    ref_phones = phonemize(voice["text"])
    tgt_phones = phonemize(text)

    prompt = build_prompt(ref_phones, tgt_phones, ref_codes)

    output = llm(
        prompt,
        max_tokens=1500,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repetition_penalty,
        stop=[SPEECH_END],
    )

    raw_tokens = output["choices"][0]["text"]
    token_list = re.findall(r"<\|[^|]+\|>", raw_tokens)
    codes = extract_codes(token_list)

    if not codes:
        return None, "No speech codes generated. Try a different voice or lower temperature."

    codes_arr = np.array(codes, dtype=np.int32)[np.newaxis, np.newaxis, :]
    wav = codec.decode_code(codes_arr)
    wav = wav[0, 0, :]

    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read(), f"Generated {len(codes)} speech codes (~{len(codes)/50:.1f}s)"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="VieNeu-TTS Twi") as demo:
    gr.Markdown("# 🗣️ VieNeu-TTS — Asante Twi\nText-to-speech for Asante Twi using a fine-tuned VieNeu-TTS model.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text (Twi)",
                placeholder="Meda wo ase paa. Onyame nhyira wo.",
                lines=3,
            )
            voice_select = gr.Dropdown(
                choices=VOICE_NAMES,
                value=VOICES.get("default_voice", VOICE_NAMES[0]),
                label="Voice",
            )
            gguf_select = gr.Dropdown(
                choices=["GGUF Q4_K_M (smaller, faster)", "GGUF Q8_0 (larger, better quality)"],
                value="GGUF Q4_K_M (smaller, faster)",
                label="Model",
            )
            with gr.Accordion("Advanced", open=False):
                temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
                top_p       = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Repetition Penalty")
            generate_btn = gr.Button("🎙️ Generate", variant="primary")

        with gr.Column():
            audio_output  = gr.Audio(label="Output", type="binary")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[text_input, voice_select, gguf_select, temperature, top_p, rep_penalty],
        outputs=[audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
