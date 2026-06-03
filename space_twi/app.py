import os
import re
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from phonemizer import phonemize as _ph
from vieneu.utils import NeuCodecOnnx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_REPO  = "michsethowusu/VieNeu-TTS-Twi"
CODEC_REPO  = "neuphonic/neucodec-onnx-decoder-int8"
SAMPLE_RATE = 24000
HF_TOKEN    = os.environ.get("HF_TOKEN") or None  # empty string -> None (public repo)
GGUF_FILES = {
    "GGUF Q4_K_M (smaller, faster)": "VieNeu-TTS-Twi-Q4_K_M.gguf",
    "GGUF Q8_0 (larger, better quality)": "VieNeu-TTS-Twi-Q8_0.gguf",
}

# ---------------------------------------------------------------------------
# Shared components. The GGUF embeds a broken tokenizer, so we tokenize with
# the ORIGINAL training tokenizer and feed token IDs straight to llama.cpp.
# ---------------------------------------------------------------------------
print("Loading tokenizer + codec ...")
TOKENIZER     = AutoTokenizer.from_pretrained(MODEL_REPO, token=HF_TOKEN, trust_remote_code=True)
SPEECH_END_ID = TOKENIZER.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
CODEC         = NeuCodecOnnx.from_pretrained(CODEC_REPO)

# ---------------------------------------------------------------------------
# Lazy GGUF weights (one per variant)
# ---------------------------------------------------------------------------
_models = {}

def get_model(gguf_file: str):
    if gguf_file not in _models:
        import multiprocessing
        from llama_cpp import Llama
        n_threads = max(multiprocessing.cpu_count(), 1)
        path = hf_hub_download(repo_id=MODEL_REPO, filename=gguf_file, token=HF_TOKEN)
        print(f"Loading GGUF weights: {gguf_file} (n_threads={n_threads}) ...")
        _models[gguf_file] = Llama(
            model_path=path,
            n_ctx=2048,
            n_gpu_layers=0,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            verbose=False,
        )
    return _models[gguf_file]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def phonemize(text: str) -> str:
    return _ph(text, backend="espeak", language="lfn", with_stress=True, preserve_punctuation=True)

def split_sentences(text: str):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return parts or [text.strip()]

def generate_codes(llm, target_phones, temperature, top_p, repetition_penalty, max_tokens=600):
    # No-voice prompt: just the target phonemes, no reference codes.
    prompt = (
        f"<|TEXT_PROMPT_START|>{target_phones.strip()}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>"
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
def generate(text, gguf_choice, temperature, top_p, repetition_penalty):
    if not text.strip():
        return None, "Please enter some text."

    llm = get_model(GGUF_FILES[gguf_choice])

    all_codes = []
    for sentence in split_sentences(text):
        codes = generate_codes(llm, phonemize(sentence), temperature, top_p, repetition_penalty)
        all_codes.extend(codes)

    if not all_codes:
        return None, "No speech generated. Try lowering temperature."

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
            gguf_select = gr.Dropdown(choices=list(GGUF_FILES.keys()),
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
        inputs=[text_input, gguf_select, temperature, top_p, rep_penalty],
        outputs=[audio_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
