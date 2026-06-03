"""
Push the fine-tuned Twi TTS model to Hugging Face Hub.

Steps:
  1. merge_and_push  — merge LoRA into base model weights, push safetensors + voices.json
  2. convert_gguf    — convert merged model to Q4_K_M GGUF, push to same repo (optional)

Usage
-----
Merge + push safetensors (required first):
  modal run finetune/push_to_hub.py --hf-repo michsethowusu/VieNeu-TTS-Twi

Also create + push GGUF (optional, adds ~15 min):
  modal run finetune/push_to_hub.py --hf-repo michsethowusu/VieNeu-TTS-Twi --do-gguf

Prerequisites
-------------
  Create a Modal secret named "huggingface-secret" with key HF_TOKEN set to
  a HuggingFace token that has write access to the target repo.
"""

from __future__ import annotations
import os
import modal

APP_NAME      = "vieneu-tts-push"
OUTPUT_MOUNT  = "/vol/output"
DATASET_MOUNT = "/vol/dataset"

out_vol  = modal.Volume.from_name("vieneu-tts-output",  create_if_missing=False)
data_vol = modal.Volume.from_name("vieneu-tts-dataset", create_if_missing=False)

VOLUMES = {
    OUTPUT_MOUNT:  out_vol,
    DATASET_MOUNT: data_vol,
}

RUN_NAME          = "VieNeu-TTS-0.3B-LoRA-twi-v4"
BASE_MODEL        = "pnnbao-ump/VieNeu-TTS-0.3B"
HF_CHECKPOINT_REPO = "michsethowusu/VieNeu-TTS-Twi-LoRA"

# Base image — same stack as training
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential", "libsndfile1")
    .pip_install(
        "torch",
        "transformers>=4.46.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "huggingface_hub>=0.23.0",
        "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

# GGUF conversion image adds llama.cpp build tools
gguf_image = (
    base_image
    .run_commands(
        "git clone --depth=1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DLLAMA_NATIVE=OFF",
        "cmake --build /opt/llama.cpp/build --config Release -j$(nproc) --target llama-quantize",
    )
    .pip_install("gguf", "numpy", "sentencepiece")
)

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Step 1 — Merge LoRA + push safetensors + voices.json
# ---------------------------------------------------------------------------

@app.function(
    image=base_image,
    volumes=VOLUMES,
    gpu="A10G",
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_and_push(hf_repo: str, voices_json_str: str = "", checkpoint: str = "final"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import HfApi, snapshot_download

    hf_token    = os.environ["HF_TOKEN"]
    merged_path = os.path.join(OUTPUT_MOUNT, RUN_NAME, "merged")

    # Resolve adapter path — download from HF if missing or incomplete
    adapter_path = os.path.join(OUTPUT_MOUNT, RUN_NAME, checkpoint)
    config_file  = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_file):
        print(f"Adapter incomplete or missing — downloading {checkpoint} from {HF_CHECKPOINT_REPO} ...")
        snapshot_download(
            repo_id=HF_CHECKPOINT_REPO,
            repo_type="model",
            token=hf_token,
            local_dir=adapter_path,
            allow_patterns=[f"{checkpoint}/*"],
        )
        # snapshot_download nests: adapter_path/checkpoint/files — unwrap if needed
        nested = os.path.join(adapter_path, checkpoint)
        if os.path.exists(os.path.join(nested, "adapter_config.json")):
            adapter_path = nested
    voices_src   = os.path.join(DATASET_MOUNT, "voices.json")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}. Run training first.")

    # --- Merge ---
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Applying LoRA from {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    print(f"Saving merged model → {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    out_vol.commit()

    # --- Push to HF ---
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)

    print(f"Uploading model files to {hf_repo} ...")
    api.upload_folder(
        folder_path=merged_path,
        repo_id=hf_repo,
        repo_type="model",
    )

    # Upload voices.json — prefer content passed from local, fallback to volume
    if not voices_json_str:
        for candidate in [
            os.path.join(OUTPUT_MOUNT, RUN_NAME, "voices.json"),
            voices_src,
        ]:
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as f:
                    voices_json_str = f.read()
                break

    if voices_json_str:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(voices_json_str)
            tmp_path = tmp.name
        print("Uploading voices.json ...")
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="voices.json",
            repo_id=hf_repo,
            repo_type="model",
        )
        os.unlink(tmp_path)
    else:
        print("WARNING: voices.json not found — upload it manually later.")

    print(f"\n✅ Model pushed to https://huggingface.co/{hf_repo}")


# ---------------------------------------------------------------------------
# Step 2 (optional) — Convert merged model to GGUF Q4_K_M + push
# ---------------------------------------------------------------------------

@app.function(
    image=gguf_image,
    volumes=VOLUMES,
    gpu=None,
    cpu=4,
    timeout=3600,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def convert_gguf(hf_repo: str):
    import subprocess
    from huggingface_hub import HfApi

    hf_token    = os.environ["HF_TOKEN"]
    merged_path = os.path.join(OUTPUT_MOUNT, RUN_NAME, "merged")
    f16_path    = os.path.join(OUTPUT_MOUNT, RUN_NAME, "model-f16.gguf")
    q4_path     = os.path.join(OUTPUT_MOUNT, RUN_NAME, "VieNeu-TTS-Twi-Q4_K_M.gguf")
    q8_path     = os.path.join(OUTPUT_MOUNT, RUN_NAME, "VieNeu-TTS-Twi-Q8_0.gguf")

    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged model not found at {merged_path}. Run merge_and_push first.")

    # Patch llama.cpp's tokenizer hash check — VieNeu-TTS uses a custom BPE
    # tokenizer (extra speech tokens) whose hash isn't in llama.cpp's registry.
    base_py = "/opt/llama.cpp/conversion/base.py"
    with open(base_py, "r") as f:
        src = f.read()
    patched = src.replace(
        'raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")',
        'logger.warning("Unknown BPE pre-tokenizer hash, falling back to gpt2"); return "gpt2"',
    )
    with open(base_py, "w") as f:
        f.write(patched)
    print("Patched llama.cpp tokenizer hash check.")

    # Convert to F16 GGUF
    print("Converting merged model to F16 GGUF ...")
    subprocess.run(
        ["python", "/opt/llama.cpp/convert_hf_to_gguf.py",
         merged_path, "--outtype", "f16", "--outfile", f16_path],
        check=True,
    )

    # Quantize to Q4_K_M (small, ~25% of original — best for CPU RAM-constrained)
    print("Quantizing to Q4_K_M ...")
    subprocess.run(
        ["/opt/llama.cpp/build/bin/llama-quantize", f16_path, q4_path, "Q4_K_M"],
        check=True,
    )

    # Quantize to Q8_0 (larger, ~50% of original — better quality, still fast on CPU)
    print("Quantizing to Q8_0 ...")
    subprocess.run(
        ["/opt/llama.cpp/build/bin/llama-quantize", f16_path, q8_path, "Q8_0"],
        check=True,
    )

    os.remove(f16_path)
    out_vol.commit()

    api = HfApi(token=hf_token)
    for local_path, repo_name in [(q4_path, "VieNeu-TTS-Twi-Q4_K_M.gguf"), (q8_path, "VieNeu-TTS-Twi-Q8_0.gguf")]:
        print(f"Uploading {repo_name} to {hf_repo} ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_name,
            repo_id=hf_repo,
            repo_type="model",
        )
    print(f"\n✅ GGUF files pushed to https://huggingface.co/{hf_repo}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    hf_repo: str    = "michsethowusu/VieNeu-TTS-Twi",
    checkpoint: str = "checkpoint-3000",   # which LoRA checkpoint to merge
    do_gguf: bool   = False,
):
    import pathlib
    vj_path = pathlib.Path(__file__).parent / "voices.json"
    voices_json_str = vj_path.read_text(encoding="utf-8") if vj_path.exists() else ""
    if voices_json_str:
        print(f"[local] Found voices.json ({len(voices_json_str)} bytes), will upload alongside model.")
    else:
        print("[local] WARNING: voices.json not found locally.")

    print(f"[local] Merging {checkpoint} and pushing safetensors to {hf_repo} ...")
    merge_and_push.remote(hf_repo, voices_json_str, checkpoint)

    if do_gguf:
        print(f"[local] Converting to GGUF (Q4_K_M + Q8_0) and pushing to {hf_repo} ...")
        convert_gguf.remote(hf_repo)
    else:
        print("[local] Skipping GGUF (pass --do-gguf to also create quantized CPU versions)")
