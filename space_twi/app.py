import os
import re
import json
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from phonemizer import phonemize as _ph
from vieneu.utils import NeuCodecOnnx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO    = "michsethowusu/VieNeu-TTS-Twi"
CODEC_REPO    = "neuphonic/neucodec-onnx-decoder-int8"
SAMPLE_RATE   = 24000
MAX_REF_CODES = 200
HF_TOKEN      = os.environ.get("HF_TOKEN") or None  # empty string -> None (public repo)
GGUF_FILES = {
    "GGUF Q4_K_M (smaller, faster)": "VieNeu-TTS-Twi-Q4_K_M.gguf",
    "GGUF Q8_0 (larger, better quality)": "VieNeu-TTS-Twi-Q8_0.gguf",
}

# ---------------------------------------------------------------------------
# Load shared components: original tokenizer, ONNX codec, voices
# (The GGUF embeds a broken tokenizer, so we tokenize with the ORIGINAL
#  training tokenizer and feed token IDs straight to llama.cpp.)
# ---------------------------------------------------------------------------
print("Loading tokenizer, codec, voices ...")
TOKENIZER     = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN, trust_remote_code=True)
SPEECH_END_ID = TOKENIZER.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
CODEC         = NeuCodecOnnx.from_pretrained(CODEC_REPO)

_vpath = hf_hub_download(repo_id=MODEL_REPO, filename="voices.json", token=HF_TOKEN)
VOICES = json.load(open(_vpath, encoding="utf-8"))
VOICE_CHOICES = list(VOICES["presets"].keys())
DEFAULT_VOICE = VOICES.get("default_voice") or VOICE_CHOICES[0]

# Cache phonemized reference text per voice
_ref_phoneme_cache = {}

# ---------------------------------------------------------------------------
# Lazy GGUF weights (one per variant)
# ---------------------------------------------------------------------------
_models = {}

def get_model(gguf_file: str):
    if gguf_file not in _models:
        from llama_cpp import Llama
        path = hf_hub_download(repo_id=MODEL_REPO, filename=gguf_file, token=HF_TOKEN)
        print(f"Loading GGUF weights: {gguf_file} ...")
        _models[gguf_file] = Llama(model_path=path, n_ctx=2048, n_gpu_layers=0, verbose=False)
    return _models[gguf_file]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def phonemize(text: str) -> str:
    return _ph(text, backend="espeak", language="lfn", with_stress=True, preserve_punctuation=True)

def split_sentences(text: str):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts or [text.strip()]

def generate_codes(llm, ref_codes, ref_phones, target_phones,
                   temperature, top_p, repetition_penalty, max_tokens=1200):
    codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes[:MAX_REF_CODES])
    prompt = (
        f"<|TEXT_PROMPT_START|>{ref_phones.strip()} {target_phones.strip()}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}"
    )
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False)

    out_ids = []
    for tid in llm.generate(prompt_ids, temp=temperature, top_p=top_p, top_k=50,
                            repeat_penalty=repetition_penalty):
        if tid == SPEECH_END_ID or tid == TOKENIZER.eos_token_id:
            break
        out_ids.append(tid)
        if len(out_ids) >= max_tokens:
            break

    toks = TOKENIZER.convert_ids_to_tokens(out_ids)
    return [int(m.group(1)) for t in toks for m in [re.match(r"<\|speech_(\d+)\|>", t or "")] if m]

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate(text, voice_name, gguf_choice, temperature, top_p, repetition_penalty):
    if not text.strip():
        return None, "Please enter some text."

    llm   = get_model(GGUF_FILES[gguf_choice])
    voice = VOICES["presets"][voice_name]
    ref_codes = voice["codes"]

    if voice_name not in _ref_phoneme_cache:
        _ref_phoneme_cache[voice_name] = phonemize(voice["text"])
    ref_phones = _ref_phoneme_cache[voice_name]

    all_codes = []
    for sentence in split_sentences(text):
        tgt_phones = phonemize(sentence)
        codes = generate_codes(llm, ref_codes, ref_phones, tgt_phones,
                               temperature, top_p, repetition_penalty)
        all_codes.extend(codes)

    if not all_codes:
        return None, "No speech generated. Try lowering temperature or a different voice."

    wav = CODEC.decode_code(np.array(all_codes, dtype=np.int32))
    wav = np.asarray(wav).reshape(-1).astype(np.float32)
    return (SAMPLE_RATE, wav), f"Generated {len(all_codes)} codes (~{len(all_codes)/50:.1f}s)."

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="VieNeu-TTS Twi") as demo:
    gr.Markdown(
        "# 🗣️ VieNeu-TTS — Asante Twi\n"
        "Text-to-speech for Asante Twi, fine-tuned from VieNeu-TTS-0.3B. Runs on CPU (GGUF)."
    )
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text (Twi)",
                                    placeholder="Meda wo ase paa. Onyame nhyira wo.", lines=3)
            voice_select = gr.Dropdown(choices=VOICE_CHOICES, value=DEFAULT_VOICE, label="Voice")
            gguf_select  = gr.Dropdown(choices=list(GGUF_FILES.keys()),
                                       value="GGUF Q4_K_M (smaller, faster)", label="Model")
            with gr.Accordion("Advanced", open=False):
                temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
                top_p       = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.3, step=0.05, label="Repetition Penalty")
            generate_btn = gr.Button("🎙️ Generate", variant="primary")
        with gr.Column():
            audio_output  = gr.Audio(label="Output", type="numpy")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[text_input, voice_select, gguf_select, temperature, top_p, rep_penalty],
        outputs=[audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
