"""
Akiti-TTS on-demand inference endpoint on Modal (CPU, scale-to-zero).

Deploy:
  modal deploy finetune/akiti_infer.py

Containers spin up on demand and scale back to zero when idle (no always-on
cost). Inference uses the GGUF weights with the ORIGINAL training tokenizer
(the GGUF's embedded tokenizer is broken), decoding via the ONNX int8 codec.
"""
import modal

APP_NAME   = "akiti-tts"
MODEL_REPO = "michsethowusu/Akiti-TTS"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
WHEEL_URL  = "https://huggingface.co/michsethowusu/Akiti-TTS/resolve/main/wheels/llama_cpp_python-0.3.25-py3-none-linux_x86_64.whl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "espeak-ng", "libsndfile1")
    .pip_install(WHEEL_URL, "huggingface_hub", "phonemizer", "numpy",
                 "onnxruntime", "transformers", "tokenizers", "soundfile")
)

app = modal.App(APP_NAME, image=image)

GGUF_FILES = {
    "q4": "VieNeu-TTS-Twi-Q4_K_M.gguf",
    "q8": "VieNeu-TTS-Twi-Q8_0.gguf",
}


@app.cls(
    image=image,
    cpu=8.0,
    memory=8192,
    scaledown_window=300,   # stay warm 5 min after last request
    min_containers=0,       # scale to zero when idle (no idle cost)
    timeout=600,
)
class Akiti:
    @modal.enter()
    def load(self):
        import re, json, multiprocessing
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from llama_cpp import Llama
        from phonemizer import phonemize as _ph

        self.re, self.np, self._ph = re, np, _ph
        self.tok = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
        self.speech_end_id = self.tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

        # ONNX int8 codec decoder (inlined — just an onnxruntime session)
        onnx_path = hf_hub_download(repo_id=CODEC_REPO, filename="model.onnx")
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.codec = onnxruntime.InferenceSession(onnx_path, sess_options=so,
                                                  providers=["CPUExecutionProvider"])

        vpath = hf_hub_download(repo_id=MODEL_REPO, filename="voices.json")
        self.voices = json.load(open(vpath, encoding="utf-8"))
        self._ref_cache = {}

        n_threads = max(multiprocessing.cpu_count(), 1)
        self.models = {}
        for key, fname in GGUF_FILES.items():
            p = hf_hub_download(repo_id=MODEL_REPO, filename=fname)
            self.models[key] = Llama(model_path=p, n_ctx=2048, n_gpu_layers=0,
                                     n_threads=n_threads, n_threads_batch=n_threads, verbose=False)
        print("Akiti models loaded.")

    def _phon(self, t):
        return self._ph(t, backend="espeak", language="lfn", with_stress=True, preserve_punctuation=True)

    def _decode(self, codes):
        arr = self.np.array(codes, dtype=self.np.int32)[None, None, :]
        recon = self.codec.run(None, {"codes": arr})[0]
        return self.np.asarray(recon).reshape(-1).astype("float32")

    def _gen_codes(self, llm, prompt, temperature, top_p, rep_pen, max_tokens=600):
        ids = self.tok.encode(prompt, add_special_tokens=False)
        out = []
        for tid in llm.generate(ids, temp=temperature, top_p=top_p, top_k=50, repeat_penalty=rep_pen):
            if tid == self.speech_end_id or tid == self.tok.eos_token_id:
                break
            out.append(tid)
            if len(out) >= max_tokens:
                break
        toks = self.tok.convert_ids_to_tokens(out)
        return [int(m.group(1)) for t in toks
                for m in [self.re.match(r"<\|speech_(\d+)\|>", t or "")] if m]

    @modal.method()
    def infer(self, text: str, voice: str = "none", model: str = "q4",
              temperature: float = 0.4, top_p: float = 0.8, rep_pen: float = 1.3) -> bytes:
        import io, soundfile as sf
        llm = self.models.get(model, self.models["q4"])

        ref_codes, ref_phones = None, None
        if voice and voice != "none" and voice in self.voices["presets"]:
            v = self.voices["presets"][voice]
            ref_codes = v["codes"][:200]
            if voice not in self._ref_cache:
                self._ref_cache[voice] = self._phon(v["text"])
            ref_phones = self._ref_cache[voice]

        all_codes = []
        sentences = [s.strip() for s in self.re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()] or [text]
        for sent in sentences:
            tgt = self._phon(sent).strip()
            if ref_codes is not None:
                codes_str = "".join(f"<|speech_{c}|>" for c in ref_codes)
                prompt = (f"<|TEXT_PROMPT_START|>{ref_phones.strip()} {tgt}"
                          f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{codes_str}")
            else:
                prompt = f"<|TEXT_PROMPT_START|>{tgt}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>"
            all_codes.extend(self._gen_codes(llm, prompt, temperature, top_p, rep_pen))

        if not all_codes:
            raise ValueError("No speech generated.")
        wav = self._decode(all_codes)
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV")
        return buf.getvalue()

    @modal.method()
    def list_voices(self) -> list:
        return list(self.voices["presets"].keys())
