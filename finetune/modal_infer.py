"""
Modal inference script for VieNeu-TTS LoRA checkpoint.

Runs on a GPU in the Modal cloud:
  1. (Optional) Encodes reference audio with NeuCodec -> voices.json
  2. Loads base model + LoRA adapter from the Modal volume
  3. Generates speech with in-context voice cloning
  4. Returns WAV bytes to the local caller -> saved as output.wav

Usage
-----
Build voices.json on Modal GPU (first time / after changing ref_audio):
  modal run finetune/modal_infer.py --build-voices --text "Meda wo ase paa"

Build voices only (skip inference):
  modal run finetune/modal_infer.py --build-voices --text ""

Default voice, checkpoint-3000:
  modal run finetune/modal_infer.py --text "Meda wo ase paa"

Pick a specific voice (twi_voice_0 through twi_voice_4):
  modal run finetune/modal_infer.py --text "..." --voice-name twi_voice_1

Use a specific checkpoint step:
  modal run finetune/modal_infer.py --text "..." --checkpoint checkpoint-2500

Save to a custom output file:
  modal run finetune/modal_infer.py --text "..." --output my_output.wav

HF_TOKEN (for private models):
  Create a Modal secret named "huggingface-secret" with key HF_TOKEN, then
  uncomment the secrets=[...] lines in the function decorators below.
"""

from __future__ import annotations

import io
import json
import os
import re

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure (reuses volumes created by modal_train.py)
# ---------------------------------------------------------------------------

APP_NAME = "vieneu-tts-infer"

DATASET_MOUNT = "/vol/dataset"
OUTPUT_MOUNT  = "/vol/output"

data_vol = modal.Volume.from_name("vieneu-tts-dataset", create_if_missing=False)
out_vol  = modal.Volume.from_name("vieneu-tts-output",  create_if_missing=False)

VOLUMES = {
    DATASET_MOUNT: data_vol,
    OUTPUT_MOUNT:  out_vol,
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "espeak-ng",
        "libsndfile1",
        "ffmpeg",
    )
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=4.46.0",
        "tokenizers>=0.21",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "phonemizer>=3.2.2",
        "soundfile",
        "librosa>=0.10.0",
        "neucodec>=0.0.4",
        "huggingface_hub>=0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App(APP_NAME, image=image)

RUN_NAME         = "VieNeu-TTS-0.3B-LoRA-twi-v4"
BASE_MODEL       = "pnnbao-ump/VieNeu-TTS-0.3B"
CODEC_REPO       = "neuphonic/neucodec"
HF_ADAPTER_REPO  = "michsethowusu/VieNeu-TTS-Twi-LoRA"
HF_DATASET_REPO  = "michsethowusu/vieneu-tts-twi-encoded"

# ---------------------------------------------------------------------------
# Remote inference function
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    gpu="A10G",
    timeout=600,
    memory=16384,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def infer_remote(
    text: str,
    voices_json_str: str = "",
    voice_name: str | None = None,
    checkpoint: str = "checkpoint-2000",
    hf_checkpoint: str = "",        # if set, load adapter from HF instead of volume
    max_new_tokens: int = 1500,
    max_ref_codes: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,  # >1.0 penalises repeated tokens (silences/padding)
    chunk: bool = True,             # split text into short phrases and stitch audio
) -> bytes:
    """Run inference on Modal GPU, return WAV bytes."""
    import numpy as np
    import soundfile as sf
    import torch
    from phonemizer import phonemize as _phonemize
    from neucodec import NeuCodec
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    def phonemize_text(t: str) -> str:
        return _phonemize(
            t,
            backend="espeak",
            language="lfn",
            with_stress=True,
            preserve_punctuation=True,
        )

    def split_chunks(t: str):
        """Split on sentence boundaries; further split long clauses on commas."""
        import re as _re
        sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]
        chunks = []
        for s in sentences:
            parts = [p.strip() for p in _re.split(r'(?<=,)\s+', s) if p.strip()]
            chunks.extend(parts)
        return chunks if chunks else [t]

    # --- load voice preset (ref text + ref codes) ---
    use_voice = bool(voices_json_str)
    if use_voice:
        voices_data = json.loads(voices_json_str)
        if voice_name is None:
            voice_name = voices_data.get("default_voice")
        if voice_name not in voices_data["presets"]:
            available = list(voices_data["presets"].keys())
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")
        voice     = voices_data["presets"][voice_name]
        ref_text  = voice["text"]
        ref_codes = voice["codes"][:max_ref_codes]
        print(f"Voice: {voice_name}  ref_codes: {len(ref_codes)}")
    else:
        ref_text  = ""
        ref_codes = []
        print("No voice — generating without reference codes.")

    # --- tokenizer + model ---
    print(f"Loading tokenizer from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load adapter — prefer HF download, fall back to volume
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_checkpoint:
        from huggingface_hub import snapshot_download
        ckpt_name    = hf_checkpoint
        local_ckpt   = f"/tmp/{ckpt_name}"
        print(f"Downloading {ckpt_name} from {HF_ADAPTER_REPO} ...")
        snapshot_download(
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
            token=hf_token or None,
            local_dir=local_ckpt,
            allow_patterns=[f"{ckpt_name}/*"],
        )
        adapter_path = os.path.join(local_ckpt, ckpt_name)
    else:
        adapter_path = os.path.join(OUTPUT_MOUNT, RUN_NAME, checkpoint)

    if os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"WARNING: adapter not found at {adapter_path}. Running base model only.")

    model.eval()

    # --- codec ---
    print(f"Loading NeuCodec ({CODEC_REPO}) ...")
    codec = NeuCodec.from_pretrained(CODEC_REPO).to(device)
    codec.eval()

    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    eos_id = speech_end_id if speech_end_id != tokenizer.unk_token_id else tokenizer.eos_token_id

    def generate_chunk(chunk_text: str, cur_ref_codes: list, cur_ref_text: str):
        """Generate waveform for one chunk. Returns (waveform, generated_codes)."""
        target_phones = phonemize_text(chunk_text)
        if use_voice:
            cur_ref_phones = phonemize_text(cur_ref_text)
            ref_codes_str  = "".join(f"<|speech_{c}|>" for c in cur_ref_codes)
            prompt_str = (
                f"<|TEXT_PROMPT_START|>{cur_ref_phones.strip()} {target_phones.strip()}<|TEXT_PROMPT_END|>"
                f"<|SPEECH_GENERATION_START|>{ref_codes_str}"
            )
        else:
            prompt_str = (
                f"<|TEXT_PROMPT_START|>{target_phones.strip()}<|TEXT_PROMPT_END|>"
                f"<|SPEECH_GENERATION_START|>"
            )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_id,
            )

        gen_ids = outputs[0][input_ids.shape[1]:].cpu().tolist()
        codes = []
        for tid in gen_ids:
            tok = tokenizer.convert_ids_to_tokens(tid)
            m = re.match(r"<\|speech_(\d+)\|>", tok or "")
            if m:
                codes.append(int(m.group(1)))

        if not codes:
            print(f"  Warning: no codes generated for chunk: {chunk_text!r}")
            return np.zeros(0, dtype=np.float32), cur_ref_codes

        codes_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            wav = codec.decode_code(codes_t)
        return wav.squeeze().cpu().numpy(), codes

    # --- chunk or single-pass ---
    chunks = split_chunks(text) if chunk else [text]
    print(f"Chunks ({len(chunks)}): {chunks}")

    # Seed the rolling ref with the original voice preset.
    # After each chunk, replace ref with the tail of what was just generated
    # so every chunk sounds like the one before it.
    rolling_ref_codes = list(ref_codes)
    rolling_ref_text  = ref_text

    wavs = []
    for i, c in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {c!r}")
        wav_chunk, generated_codes = generate_chunk(c, rolling_ref_codes, rolling_ref_text)
        wavs.append(wav_chunk)
        if generated_codes:
            rolling_ref_codes = generated_codes[-max_ref_codes:]
            rolling_ref_text  = c

    wav_np = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
    total_s = len(wav_np) / 24000
    print(f"Total audio: {total_s:.1f}s")

    # --- return as WAV bytes ---
    buf = io.BytesIO()
    sf.write(buf, wav_np, 24000, format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Build voices from the encoded training dataset (no GPU needed)
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    timeout=120,
    memory=2048,
)
def build_voices_from_dataset(
    n_voices: int = 5,
    min_codes: int = 200,   # must have at least enough for max_ref_codes
    max_codes: int = 0,     # 0 = no upper limit
) -> str:
    """Pick n clean samples from the encoded training dataset and return voices.json."""
    import json
    import random

    encoded_csv = os.path.join(DATASET_MOUNT, "metadata_encoded.csv")
    if not os.path.exists(encoded_csv):
        raise FileNotFoundError(f"Encoded dataset not found at {encoded_csv}. Run encode_data first.")

    # Load all samples, keep only those with enough codes to carry voice identity
    candidates = []
    with open(encoded_csv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            filename, text, codes_str = parts[0], parts[1], parts[2]
            try:
                codes = json.loads(codes_str)
            except Exception:
                continue
            if len(codes) >= min_codes and (max_codes == 0 or len(codes) <= max_codes):
                candidates.append({"filename": filename, "text": text, "codes": codes})

    if not candidates:
        raise RuntimeError(f"No samples found with >= {min_codes} codes in the dataset.")

    print(f"Found {len(candidates)} candidate samples, picking {n_voices} ...")
    # Space picks evenly across the dataset so we get variety
    step = max(1, len(candidates) // n_voices)
    picks = [candidates[i * step] for i in range(n_voices)]

    voices_data = {
        "meta": {
            "spec": "vieneu.voice.presets",
            "spec_version": "1.0",
            "engine": "VieNeu-TTS",
        },
        "default_voice": None,
        "presets": {},
    }

    for idx, sample in enumerate(picks):
        voice_name = f"twi_voice_{idx}"
        voices_data["presets"][voice_name] = {
            "codes": sample["codes"],
            "text":  sample["text"],
            "description": f"Training sample: {sample['filename']}",
        }
        if idx == 0:
            voices_data["default_voice"] = voice_name
        print(f"  {voice_name}: {sample['filename']} ({len(sample['codes'])} codes) | {sample['text'][:50]}")

    return json.dumps(voices_data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Build voices on Modal GPU
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=300,
    memory=8192,
)
def build_voices_remote(
    audio_files: dict,    # {filename: bytes}
    metadata_lines: list, # ["fname|text", ...]
) -> str:
    """Encode reference audios with NeuCodec on GPU, return voices.json as a string."""
    import io
    import json
    import librosa
    import torch
    from neucodec import NeuCodec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NeuCodec on {device} ...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()

    voices_data = {
        "meta": {
            "spec": "vieneu.voice.presets",
            "spec_version": "1.0",
            "engine": "VieNeu-TTS",
            "author": "Phạm Nguyễn Ngọc Bảo (pnnbao-ump)",
            "license": "CC BY-NC 4.0",
            "homepage": "https://github.com/pnnbao97/VieNeu-TTS",
            "notice": "Model and voices are for non-commercial use only. Mention pnnbao-ump when using.",
        },
        "default_voice": None,
        "presets": {},
    }

    for idx, line in enumerate(metadata_lines):
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue
        fname, text = parts
        voice_name = f"twi_voice_{idx}"

        audio_bytes = audio_files[fname]
        wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            codes = codec.encode_code(wav_t).squeeze(0).squeeze(0)
        codes_list = codes.cpu().numpy().flatten().tolist()

        voices_data["presets"][voice_name] = {
            "codes": codes_list,
            "text": text,
            "description": f"Twi dataset reference {idx}: {fname}",
        }
        if idx == 0:
            voices_data["default_voice"] = voice_name

        print(f"  {fname} -> {voice_name} ({len(codes_list)} codes)")

    return json.dumps(voices_data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def _resolve_latest_hf_checkpoint(repo_id: str, token: str = "") -> str:
    """Return the checkpoint folder name with the highest step number from an HF repo."""
    import re
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id, repo_type="model", token=token or None)
    nums = set()
    for f in files:
        m = re.match(r"checkpoint-(\d+)/", f)
        if m:
            nums.add(int(m.group(1)))
    if not nums:
        raise RuntimeError(f"No checkpoint folders found in {repo_id}")
    latest = max(nums)
    print(f"[local] Latest checkpoint in {repo_id}: checkpoint-{latest}")
    return f"checkpoint-{latest}"


@app.local_entrypoint()
def main(
    text: str = "Meda wo ase paa",
    voice_name: str = "",
    checkpoint: str = "checkpoint-2000",
    hf_checkpoint: str = "",   # "latest" to auto-detect, or e.g. "checkpoint-3000"
    output: str = "output.wav",
    max_new_tokens: int = 1500,
    max_ref_codes: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    voices_json: str = "",
    no_voice: bool = False,
    ref_audio: str = "",          # path to a single external wav file
    ref_audio_text: str = "",     # transcript of the external wav (optional)
    ref_audio_dir: str = "",
    build_voices: bool = False,
    from_dataset: bool = False,
    chunk: bool = True,           # False = single-pass, no splitting
    repetition_penalty: float = 1.0,
):
    import pathlib

    script_dir = pathlib.Path(__file__).parent
    vj_path    = pathlib.Path(voices_json) if voices_json else script_dir / "voices.json"

    # Resolve "latest" to the actual highest checkpoint in the HF repo
    if hf_checkpoint == "latest":
        hf_token = os.environ.get("HF_TOKEN", "")
        hf_checkpoint = _resolve_latest_hf_checkpoint(HF_ADAPTER_REPO, hf_token)

    # ------------------------------------------------------------------
    # Step 0 (optional): encode a single external wav -> use as ref
    # ------------------------------------------------------------------
    if ref_audio:
        import json as _json
        wav_path = pathlib.Path(ref_audio)
        if not wav_path.exists():
            raise FileNotFoundError(f"ref_audio not found: {wav_path}")
        fname    = wav_path.name
        metadata = [f"{fname}|{ref_audio_text or 'reference audio'}"]
        print(f"[local] Encoding external ref audio: {wav_path} ...")
        voices_json_str = build_voices_remote.remote(
            {fname: wav_path.read_bytes()}, metadata
        )
        # use inline, don't overwrite voices.json
        print("[local] External ref audio encoded.")

    # ------------------------------------------------------------------
    # Step 1 (optional): encode ref audios on Modal GPU -> voices.json
    # ------------------------------------------------------------------
    elif build_voices:
        ref_dir   = pathlib.Path(ref_audio_dir) if ref_audio_dir else script_dir / "ref_audio"
        meta_file = ref_dir / "metadata.txt"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"metadata.txt not found in {ref_dir}. "
                "Make sure ref_audio/ contains the wav files and metadata.txt."
            )

        metadata_lines = [
            l.strip() for l in meta_file.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        audio_files = {}
        for line in metadata_lines:
            fname = line.split("|", 1)[0]
            wav_path = ref_dir / fname
            if not wav_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {wav_path}")
            audio_files[fname] = wav_path.read_bytes()

        print(f"[local] Building voices from {len(audio_files)} reference audios on Modal GPU ...")
        voices_json_str = build_voices_remote.remote(audio_files, metadata_lines)
        vj_path.write_text(voices_json_str, encoding="utf-8")
        print(f"[local] voices.json saved -> {vj_path.resolve()}")
    elif from_dataset:
        print("[local] Building voices.json from training dataset samples on Modal ...")
        voices_json_str = build_voices_from_dataset.remote()
        vj_path.write_text(voices_json_str, encoding="utf-8")
        print(f"[local] voices.json saved -> {vj_path.resolve()}")
    elif no_voice:
        voices_json_str = ""
    else:
        if not vj_path.exists():
            raise FileNotFoundError(
                f"voices.json not found at {vj_path}. "
                "Run with --build-voices-from-dataset to pull samples from the training data."
            )
        voices_json_str = vj_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 2: run inference (skip when text is empty)
    # ------------------------------------------------------------------
    if not text:
        print("[local] No --text provided, skipping inference.")
        return

    # For external ref audio the voice key is always twi_voice_0
    effective_voice = voice_name or ("twi_voice_0" if ref_audio else None)

    # Auto-detect chunking: use rolling chunk only when text has multiple sentences.
    # --no-chunk always forces single pass regardless.
    import re as _re
    is_multi_sentence = bool(_re.search(r'[.!?]\s+\S', text.strip()))
    effective_chunk = is_multi_sentence if chunk else False
    print(f"[local] Multi-sentence: {is_multi_sentence} → {'chunked+rolling' if effective_chunk else 'single pass'}")
    print(f"[local] Sending text to Modal: {text!r}  Voice: {effective_voice or 'default'}  Checkpoint: {checkpoint}")

    wav_bytes = infer_remote.remote(
        text=text,
        voices_json_str=voices_json_str,
        voice_name=effective_voice,
        checkpoint=checkpoint,
        hf_checkpoint=hf_checkpoint,
        max_new_tokens=max_new_tokens,
        max_ref_codes=max_ref_codes,
        temperature=temperature,
        top_p=top_p,
        chunk=effective_chunk,
        repetition_penalty=repetition_penalty,
    )

    out_path = pathlib.Path(output)
    out_path.write_bytes(wav_bytes)
    print(f"[local] Saved -> {out_path.resolve()}  ({len(wav_bytes)/1024:.1f} KB)")
