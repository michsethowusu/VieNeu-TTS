"""
Test the HYBRID approach: original HF tokenizer + llama.cpp GGUF weights.
Bypasses llama.cpp's broken embedded tokenizer by feeding token IDs directly.

  modal run finetune/test_gguf.py
"""
import modal

APP_NAME = "test-gguf-hybrid"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "espeak-ng", "libsndfile1", "git")
    .pip_install(
        "https://huggingface.co/michsethowusu/VieNeu-TTS-Twi/resolve/main/wheels/llama_cpp_python-0.3.25-py3-none-linux_x86_64.whl",
        "huggingface_hub", "phonemizer", "numpy", "onnxruntime",
        "transformers", "tokenizers",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/michsethowusu/VieNeu-TTS.git",
        "mkdir -p /usr/local/lib/python3.11/site-packages/sea_g2p",
        "printf 'class SEAPipeline:\\n    def __init__(self,*a,**k): pass\\n    def run(self,t): return t\\nclass G2P:\\n    def __init__(self,*a,**k): pass\\n    def phonemize_batch(self,t,**k): return list(t)\\nclass Normalizer:\\n    def normalize(self,t): return t\\n' > /usr/local/lib/python3.11/site-packages/sea_g2p/__init__.py",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(cpu=4, memory=8192, timeout=900)
def test_hybrid(gguf_file: str = "VieNeu-TTS-Twi-Q4_K_M.gguf"):
    import re, json
    import numpy as np
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from llama_cpp import Llama
    from phonemizer import phonemize as _ph

    REPO = "michsethowusu/VieNeu-TTS-Twi"

    def phon(t):
        return _ph(t, backend="espeak", language="lfn", with_stress=True, preserve_punctuation=True)

    # 1) Original training tokenizer (correct)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    speech_end_id = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    print(f"HF tokenizer vocab size: {len(tok)}, SPEECH_GENERATION_END id: {speech_end_id}")

    # 2) GGUF weights only
    gguf_path = hf_hub_download(repo_id=REPO, filename=gguf_file)
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_gpu_layers=0, verbose=False)

    # 3) Build prompt with a real voice
    vpath = hf_hub_download(repo_id=REPO, filename="voices.json")
    voices = json.load(open(vpath, encoding="utf-8"))
    vname = voices.get("default_voice") or list(voices["presets"].keys())[0]
    voice = voices["presets"][vname]
    ref_codes = voice["codes"][:200]
    codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
    prompt = (
        f"<|TEXT_PROMPT_START|>{phon(voice['text']).strip()} {phon('Meda wo ase paa').strip()}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}"
    )

    # 4) Encode with HF tokenizer, generate on token IDs (bypass llama.cpp tokenizer)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    print(f"Prompt token count: {len(prompt_ids)}")

    out_ids = []
    for tid in llm.generate(prompt_ids, temp=0.4, top_p=0.8, top_k=50, repeat_penalty=1.3):
        if tid == speech_end_id or tid == tok.eos_token_id:
            break
        out_ids.append(tid)
        if len(out_ids) >= 800:
            break

    # 5) Decode generated IDs -> extract speech codes
    gen_tokens = tok.convert_ids_to_tokens(out_ids)
    codes = [int(m.group(1)) for t in gen_tokens for m in [re.match(r"<\|speech_(\d+)\|>", t or "")] if m]

    # 6) Decode codes -> audio via ONNX codec (full pipeline, like the Space)
    from vieneu.utils import NeuCodecOnnx
    codec = NeuCodecOnnx.from_pretrained("neuphonic/neucodec-onnx-decoder-int8")
    wav = np.asarray(codec.decode_code(np.array(codes, dtype=np.int32))).reshape(-1)

    result = {
        "voice": vname,
        "speech_codes": len(codes),
        "wav_samples": int(wav.shape[0]),
        "duration_s": round(wav.shape[0] / 24000, 2),
        "wav_ok": bool(np.isfinite(wav).all() and wav.shape[0] > 0),
    }
    print("\n" + "=" * 50)
    print("RESULT:", result)
    print("=" * 50)
    return result


@app.local_entrypoint()
def main(gguf_file: str = "VieNeu-TTS-Twi-Q4_K_M.gguf"):
    result = test_hybrid.remote(gguf_file)
    print("\n>>> FINAL RESULT <<<")
    print(result)
