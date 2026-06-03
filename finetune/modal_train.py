"""
Modal script for fine-tuning VieNeu-TTS with LoRA.

Phonemization uses `phonemizer` with the espeak-ng backend and the 'lfn' language
code, which maps to Twi-compatible phoneme outputs.

Pipeline steps (run in order, each skippable via CLI flags):
  1. download_data  — pull audio + transcripts from a HuggingFace dataset
  2. filter_data    — remove out-of-range durations and bad text
  3. encode_data    — encode audio with NeuCodec  [GPU]
  4. train          — LoRA fine-tune the LLM backbone  [GPU]
  5. merge_lora     — (opt) merge LoRA adapter into the base model  [GPU]
  6. create_voices  — (opt) create voices.json from a reference audio  [GPU]

Usage
-----
Full run with defaults:
  modal run finetune/modal_train.py

Resume from encoding (data already downloaded + filtered):
  modal run finetune/modal_train.py --skip-download --skip-filter

Only run training (data + encoded CSV already in the volume):
  modal run finetune/modal_train.py --skip-download --skip-filter --skip-encode

Push already-encoded CSV to HF without re-encoding (first run in a new workspace):
  modal run finetune/modal_train.py --skip-download --skip-filter --skip-encode --skip-train --push-encoded

Train in a new workspace (pulls encoded CSV + latest checkpoint from HF automatically):
  modal run finetune/modal_train.py --skip-download --skip-filter --skip-encode

Merge after training:
  modal run finetune/modal_train.py \\
      --skip-download --skip-filter --skip-encode --skip-train --do-merge

Custom HuggingFace dataset:
  modal run finetune/modal_train.py --hf-dataset your-org/your-dataset --num-samples 5000

HuggingFace Hub targets (auto-configured):
  Encoded dataset : michsethowusu/vieneu-tts-twi-encoded  (HF dataset repo)
  Checkpoints     : michsethowusu/VieNeu-TTS-Twi-LoRA     (HF model repo)

HuggingFace token:
  Create a Modal secret named "huggingface-secret" with key HF_TOKEN set to
  a token with read+write access. encode_data and train use it automatically.
"""

from __future__ import annotations

import io
import json
import os
import random
import re

import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

APP_NAME = "vieneu-tts-finetune"

DATASET_MOUNT = "/vol/dataset"
OUTPUT_MOUNT  = "/vol/output"

# Persistent volumes — created automatically on first run
data_vol = modal.Volume.from_name("vieneu-tts-dataset", create_if_missing=True)
out_vol  = modal.Volume.from_name("vieneu-tts-output",  create_if_missing=True)

VOLUMES = {
    DATASET_MOUNT: data_vol,
    OUTPUT_MOUNT:  out_vol,
}

# Container image: PyTorch 2.4 + CUDA 12.1 base, then add ML stack and espeak
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "espeak-ng",        # runtime binary required by phonemizer
        "libsndfile1",      # soundfile I/O
        "ffmpeg",           # audio format conversion
    )
    # Single pip call — no torch version pin so neucodec's torch>=2.5.1 can resolve.
    # cu124 index ensures pip picks CUDA wheels over CPU-only ones.
    .pip_install(
        "torch",
        "torchaudio",
        "transformers>=4.46.0",
        "tokenizers>=0.21",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "phonemizer>=3.2.2",
        "datasets>=3.2.0",
        "soundfile",
        "librosa>=0.10.0",
        "tqdm",
        "neucodec>=0.0.4",
        "huggingface_hub>=0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App(APP_NAME, image=image)

# ---------------------------------------------------------------------------
# Shared configuration (inlined so the script is self-contained)
# ---------------------------------------------------------------------------

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

TRAINING_CONFIG = dict(
    model="pnnbao-ump/VieNeu-TTS-0.3B",
    run_name="VieNeu-TTS-0.3B-LoRA-twi-v4",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=10000,
    logging_steps=50,
    save_steps=500,
    warmup_ratio=0.05,
    bf16=True,
)

# HuggingFace Hub targets
HF_USERNAME             = "michsethowusu"
HF_ENCODED_DATASET_REPO = f"{HF_USERNAME}/vieneu-tts-twi-encoded"   # dataset repo for the encoded CSV
HF_CHECKPOINT_REPO      = f"{HF_USERNAME}/VieNeu-TTS-Twi-LoRA"      # model repo for checkpoints

# Regex patterns used by filter_data (module-level so the container import picks them up)
_ACRONYM           = re.compile(r"(?:[a-zA-Z]\.){2,}")
_ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){2,}")


# ---------------------------------------------------------------------------
# Internal helper — push encoded CSV to HF (not a Modal function)
# ---------------------------------------------------------------------------

def _push_encoded_to_hf(csv_path: str, repo: str = "") -> None:
    from huggingface_hub import HfApi
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("HF_TOKEN not set — skipping push of encoded dataset to HuggingFace.")
        return
    target = repo or HF_ENCODED_DATASET_REPO
    print(f"Pushing encoded dataset to {target} ...")
    api = HfApi(token=hf_token)
    api.create_repo(target, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo="metadata_encoded.csv",
        repo_id=target,
        repo_type="dataset",
        commit_message="Upload encoded TTS dataset",
    )
    print(f"Pushed → https://huggingface.co/datasets/{target}")


# ---------------------------------------------------------------------------
# Step 1 — Download data from HuggingFace
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    timeout=7200,
    memory=8192,
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # uncomment for private repos
)
def download_data(hf_dataset: str = "ghananlpcommunity/navigation-corpus-twi-speech", num_samples: int = 20000):
    import soundfile as sf
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    raw_audio_dir = os.path.join(DATASET_MOUNT, "raw_audio")
    metadata_path = os.path.join(DATASET_MOUNT, "metadata.csv")
    os.makedirs(raw_audio_dir, exist_ok=True)

    print(f"Downloading up to {num_samples} samples from {hf_dataset} ...")
    dataset = load_dataset(hf_dataset, split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    lines, count = [], 0
    for sample in tqdm(dataset, total=num_samples):
        if count >= num_samples:
            break
        try:
            audio_bytes = sample["audio"]["bytes"]
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            text     = sample["text"]
            filename = os.path.basename(sample.get("file_name", f"sample_{count:05d}.wav"))
            sf.write(os.path.join(raw_audio_dir, filename), audio_array, sr)
            lines.append(f"{filename}|{text}\n")
            count += 1
        except Exception as exc:
            print(f"  Warning – skipping sample {count}: {exc}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    data_vol.commit()
    print(f"Saved {count} samples → {DATASET_MOUNT}")


# ---------------------------------------------------------------------------
# Step 2 — Filter / clean the dataset
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    timeout=1800,
)
def filter_data():
    from tqdm import tqdm

    metadata_path = os.path.join(DATASET_MOUNT, "metadata.csv")
    cleaned_path  = os.path.join(DATASET_MOUNT, "metadata_cleaned.csv")
    raw_audio_dir = os.path.join(DATASET_MOUNT, "raw_audio")

    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    valid, skipped = [], {"no_audio": 0, "empty_text": 0}

    for line in tqdm(lines, desc="Filtering", unit="samples"):
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        filename, text = parts[0], parts[1]
        if not text.strip():
            skipped["empty_text"] += 1
            continue
        if not os.path.exists(os.path.join(raw_audio_dir, filename)):
            skipped["no_audio"] += 1
            continue
        valid.append(f"{filename}|{text}\n")

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.writelines(valid)

    data_vol.commit()
    print(f"Filter complete: {len(valid)}/{len(lines)} kept  |  skipped={skipped}")


# ---------------------------------------------------------------------------
# Step 3 — Encode audio with NeuCodec  [GPU]
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    gpu="A10G",
    cpu=8.0,
    timeout=21600,
    memory=24576,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def encode_data(max_samples: int = 20000, hf_encoded_repo: str = ""):
    import librosa
    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from neucodec import NeuCodec
    from tqdm import tqdm

    meta_clean    = os.path.join(DATASET_MOUNT, "metadata_cleaned.csv")
    meta_raw      = os.path.join(DATASET_MOUNT, "metadata.csv")
    meta_path     = meta_clean if os.path.exists(meta_clean) else meta_raw
    output_path   = os.path.join(DATASET_MOUNT, "metadata_encoded.csv")
    raw_audio_dir = os.path.join(DATASET_MOUNT, "raw_audio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NeuCodec on {device} ...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()

    with open(meta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    lines = lines[:max_samples]

    # Resume state lives in a separate progress file keyed to THIS metadata source,
    # so switching datasets never contaminates the output CSV with old data.
    progress_path = output_path + ".progress"

    already_done = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            for line in f:
                fname = line.strip()
                if fname:
                    already_done.add(fname)
        print(f"Resuming — {len(already_done)} samples already encoded, skipping them.")

    # Build work list, excluding already-encoded files
    work = []
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        filename, text = parts[0], parts[1]
        if filename in already_done:
            continue
        audio_path = os.path.join(raw_audio_dir, filename)
        if os.path.exists(audio_path):
            work.append((filename, text, audio_path))

    print(f"Encoding {len(work)} samples with {min(8, len(work))} loader threads ...")

    def load_audio(item):
        filename, text, audio_path = item
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        return filename, text, wav

    # Output CSV is always written fresh (open "w" once, then append within this run).
    # Progress file tracks completed filenames so we can resume after a crash.
    encoded, skipped = 0, 0
    open_mode = "w" if not already_done else "a"
    with open(output_path, open_mode, encoding="utf-8") as out_f, \
         open(progress_path, "a", encoding="utf-8") as prog_f:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(load_audio, item): item for item in work}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Encoding"):
                try:
                    filename, text, wav = future.result()
                    wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        codes = codec.encode_code(wav_t)
                        codes = codes.squeeze(0).squeeze(0).cpu().numpy().flatten().tolist()
                        codes = [int(c) for c in codes]
                    if not codes or not all(0 <= c < 65536 for c in codes):
                        skipped += 1
                        continue
                    out_f.write(f"{filename}|{text}|{json.dumps(codes)}\n")
                    out_f.flush()
                    prog_f.write(f"{filename}\n")
                    prog_f.flush()
                    encoded += 1
                    # Commit to volume every 200 samples so kills don't lose progress
                    if encoded % 200 == 0:
                        data_vol.commit()
                        print(f"  Checkpoint: {encoded + len(already_done)} samples committed to volume.")
                except Exception as exc:
                    print(f"  Error: {exc}")
                    skipped += 1

    # Clean up progress file on successful completion
    if os.path.exists(progress_path):
        os.remove(progress_path)

    data_vol.commit()
    print(f"Encoded {encoded} new + {len(already_done)} resumed = {encoded + len(already_done)} total, {skipped} errors")

    _push_encoded_to_hf(output_path, hf_encoded_repo)


# ---------------------------------------------------------------------------
# Push encoded data to HF (run independently when you already have the CSV)
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    timeout=600,
    memory=2048,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push_encoded_data(hf_encoded_repo: str = ""):
    """Push an already-encoded metadata_encoded.csv to HuggingFace without re-encoding."""
    output_path = os.path.join(DATASET_MOUNT, "metadata_encoded.csv")
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Encoded dataset not found at {output_path}. Run encode_data first."
        )
    _push_encoded_to_hf(output_path, hf_encoded_repo)


# ---------------------------------------------------------------------------
# Step 4 — LoRA fine-tuning  [GPU]
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    gpu="A100-80GB",
    timeout=86400,       # 24 h ceiling
    memory=65536,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(hf_encoded_repo: str = ""):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from phonemizer import phonemize as _ph
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from huggingface_hub import hf_hub_download, snapshot_download

    hf_token = os.environ.get("HF_TOKEN", "")
    encoded_repo = hf_encoded_repo or HF_ENCODED_DATASET_REPO

    # ── Pull encoded dataset from HF if not already on the volume ──────────
    encoded_csv = os.path.join(DATASET_MOUNT, "metadata_encoded.csv")
    if not os.path.exists(encoded_csv):
        if not hf_token:
            raise FileNotFoundError(
                f"Encoded dataset not found at {encoded_csv} and HF_TOKEN is not set."
            )
        print(f"Encoded dataset not found locally — downloading from {encoded_repo} ...")
        hf_hub_download(
            repo_id=encoded_repo,
            filename="metadata_encoded.csv",
            repo_type="dataset",
            token=hf_token,
            local_dir=DATASET_MOUNT,
        )
        data_vol.commit()
        print("Encoded dataset downloaded from HuggingFace.")

    # Phonemization: espeak-ng backend, 'lfn' language (Twi-compatible output)
    def phonemize_text(text: str) -> str:
        return _ph(
            text,
            backend="espeak",
            language="lfn",
            with_stress=True,
            preserve_punctuation=True,
        )

    MAX_REF_CODES = 200  # ref codes kept in prompt; rest truncated

    def preprocess_sample(target, ref, tokenizer, max_len: int = 8024):
        """In-context voice cloning format: ref + target phones in TEXT,
        ref codes + target codes in SPEECH. Labels supervised from SPEECH_START."""
        speech_start_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        ignore_index    = -100

        ref_codes_trunc  = ref["codes"][:MAX_REF_CODES]
        combined_phones  = ref["phones"].strip() + " " + target["phones"].strip()
        ref_codes_str    = "".join(f"<|speech_{c}|>" for c in ref_codes_trunc)
        target_codes_str = "".join(f"<|speech_{c}|>" for c in target["codes"])

        chat = (
            f"<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>{ref_codes_str}{target_codes_str}<|SPEECH_GENERATION_END|>"
        )

        ids = tokenizer.encode(chat)
        if len(ids) < max_len:
            ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels    = torch.full_like(input_ids, ignore_index)
        starts    = (input_ids == speech_start_id).nonzero(as_tuple=True)[0]
        if len(starts):
            labels[starts[0]:] = input_ids[starts[0]:]

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
        }

    class TTSDataset(Dataset):
        def __init__(self, metadata_path: str, tokenizer, max_len: int = 2048):
            self.tokenizer = tokenizer
            self.max_len   = max_len
            self.samples   = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        self.samples.append({
                            "filename": parts[0],
                            "text":     parts[1],
                            "codes":    json.loads(parts[2]),
                        })
            print(f"Loaded {len(self.samples)} samples — pre-phonemizing (batch) ...")
            texts = [s["text"] for s in self.samples]
            try:
                phones_list = _ph(
                    texts,
                    backend="espeak",
                    language="lfn",
                    with_stress=True,
                    preserve_punctuation=True,
                    njobs=1,
                )
                for s, p in zip(self.samples, phones_list):
                    s["phones"] = p
            except Exception as e:
                print(f"  Batch phonemization failed ({e}), falling back per-sample ...")
                for s in self.samples:
                    try:
                        s["phones"] = phonemize_text(s["text"])
                    except Exception:
                        s["phones"] = s["text"]
            print("Pre-phonemization complete.")

            # Filter out samples whose tokenized length would exceed max_len.
            # We estimate by pairing each sample with itself as the ref (combined_phones
            # = 2× its own phones), which mirrors the worst case during training.
            before = len(self.samples)
            kept = []
            for s in self.samples:
                ref_codes_trunc  = s["codes"][:MAX_REF_CODES]
                combined_phones  = s["phones"].strip() + " " + s["phones"].strip()
                ref_codes_str    = "".join(f"<|speech_{c}|>" for c in ref_codes_trunc)
                target_codes_str = "".join(f"<|speech_{c}|>" for c in s["codes"])
                chat = (
                    f"<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>"
                    f"<|SPEECH_GENERATION_START|>{ref_codes_str}{target_codes_str}"
                    f"<|SPEECH_GENERATION_END|>"
                )
                if len(tokenizer.encode(chat)) <= max_len:
                    kept.append(s)
            self.samples = kept
            print(f"Length filter: kept {len(self.samples)}/{before} samples within {max_len} tokens.")

            # Pre-compute the ref-eligible pool:
            #   - code length 100–300 (≈2–6 s, matching inference build_voices_from_dataset)
            #   - text has no punctuation (punctuation in ref text adds no useful signal
            #     and mismatches the clean-text targets the model is asked to generate)
            import string as _string
            _PUNCT = set(_string.punctuation)
            MIN_REF_CODES, MAX_REF_CODES_LEN = 100, 300
            self._ref_indices = [
                i for i, s in enumerate(self.samples)
                if MIN_REF_CODES <= len(s["codes"]) <= MAX_REF_CODES_LEN
                and not any(ch in _PUNCT for ch in s["text"])
            ]
            print(
                f"Ref pool: {len(self._ref_indices)} samples with 100–300 codes and no punctuation. "
                f"({len(self.samples) - len(self._ref_indices)} excluded)"
            )
            if not self._ref_indices:
                # Fallback to length-only filter so training doesn't crash
                self._ref_indices = [
                    i for i, s in enumerate(self.samples)
                    if MIN_REF_CODES <= len(s["codes"]) <= MAX_REF_CODES_LEN
                ]
                print("Warning: no punct-free samples in range — falling back to length-only filter.")
            if not self._ref_indices:
                self._ref_indices = list(range(len(self.samples)))
                print("Warning: no samples in 100–300 code range at all — using full pool as fallback.")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            target = self.samples[idx]
            # Pick a random ref from the 2–6 s eligible pool, excluding the target itself
            candidates = [i for i in self._ref_indices if i != idx]
            if not candidates:
                candidates = [i for i in range(len(self.samples)) if i != idx]
            ref = self.samples[random.choice(candidates)]
            return preprocess_sample(
                target,
                ref,
                self.tokenizer,
                self.max_len,
            )

    # Load model + tokenizer
    model_name = TRAINING_CONFIG["model"]
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Dataset — path is guaranteed to exist at this point
    dataset_path = os.path.join(DATASET_MOUNT, "metadata_encoded.csv")
    dataset = TTSDataset(dataset_path, tokenizer)

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    run_name   = TRAINING_CONFIG["run_name"]
    output_dir = os.path.join(OUTPUT_MOUNT, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Checkpoint resume: local first, then HF ─────────────────────────────
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            print(f"Resuming from local checkpoint: {last_checkpoint}")

    if last_checkpoint is None and hf_token:
        print(f"No local checkpoint — attempting to restore from {HF_CHECKPOINT_REPO} ...")
        try:
            snapshot_download(
                repo_id=HF_CHECKPOINT_REPO,
                repo_type="model",
                token=hf_token,
                local_dir=output_dir,
                ignore_patterns=["*.gguf"],
            )
            out_vol.commit()
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint:
                print(f"Resuming from HF checkpoint: {last_checkpoint}")
            else:
                print("HF repo exists but no trainer checkpoint found — starting fresh.")
        except Exception as exc:
            print(f"Could not restore from HF: {exc}. Starting fresh.")

    # ── Hub push kwargs (only when token is available) ───────────────────────
    hub_kwargs = {}
    if hf_token:
        hub_kwargs = dict(
            push_to_hub=True,
            hub_model_id=HF_CHECKPOINT_REPO,
            hub_token=hf_token,
            hub_strategy="all_checkpoints",
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=False,
        max_steps=TRAINING_CONFIG["max_steps"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        bf16=TRAINING_CONFIG["bf16"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        **hub_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
    )

    print("Starting LoRA training ...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_path = os.path.join(output_dir, "final")
    print(f"Saving LoRA adapter → {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    out_vol.commit()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Step 5 (optional) — Merge LoRA adapter into the base model  [GPU]
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    gpu="A10G",
    timeout=3600,
    memory=32768,
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_lora(base_model: str = "pnnbao-ump/VieNeu-TTS-0.3B"):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_name     = TRAINING_CONFIG["run_name"]
    adapter_path = os.path.join(OUTPUT_MOUNT, run_name, "final")
    merged_path  = os.path.join(OUTPUT_MOUNT, run_name, "merged")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}. Run train first.")

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model → {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    out_vol.commit()
    print("Merge complete.")


# ---------------------------------------------------------------------------
# Step 6 (optional) — Create voices.json from a reference audio  [GPU]
# ---------------------------------------------------------------------------

@app.function(
    volumes=VOLUMES,
    gpu="A10G",
    timeout=1800,
    memory=16384,
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def create_voices(
    audio_path: str,
    text: str,
    voice_name: str = "default",
    description: str = "",
    output_path: str = "",
):
    """
    Encode a reference audio and produce a voices.json preset file.

    audio_path  — path to the .wav file *inside the volume* (e.g. /vol/dataset/raw_audio/ref.wav)
    text        — exact transcript of the reference audio
    voice_name  — key used in voices.json presets
    output_path — destination inside the output volume (defaults to <OUTPUT_MOUNT>/<run_name>/voices.json)
    """
    import librosa
    import torch
    from neucodec import DistillNeuCodec

    if not output_path:
        output_path = os.path.join(OUTPUT_MOUNT, TRAINING_CONFIG["run_name"], "voices.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DistillNeuCodec on {device} ...")
    codec = DistillNeuCodec.from_pretrained("neuphonic/distill-neucodec").to(device)
    codec.eval()

    print(f"Encoding reference audio: {audio_path}")
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    wav_t  = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_codes = codec.encode_code(audio_or_path=wav_t).squeeze(0).squeeze(0)

    codes_list = ref_codes.cpu().numpy().flatten().tolist()

    # Load existing file if present, otherwise start fresh
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            voices_data = json.load(f)
    else:
        voices_data = {
            "meta": {
                "spec":         "vieneu.voice.presets",
                "spec_version": "1.0",
                "engine":       "VieNeu-TTS",
            },
            "default_voice": voice_name,
            "presets":        {},
        }

    voices_data["presets"][voice_name] = {
        "codes":       codes_list,
        "text":        text,
        "description": description or f"Custom voice: {voice_name}",
    }
    voices_data["default_voice"] = voice_name

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(voices_data, f, ensure_ascii=False, indent=2)

    out_vol.commit()
    print(f"voices.json saved → {output_path}")


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrate the full pipeline
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    hf_dataset: str       = "ghananlpcommunity/navigation-corpus-twi-speech",
    num_samples: int      = 20000,
    max_encode: int       = 20000,
    hf_encoded_repo: str  = "",   # HF dataset repo for encoded CSV; defaults to HF_ENCODED_DATASET_REPO
    skip_download: bool = False,
    skip_filter:   bool = False,
    skip_encode:   bool = False,
    skip_train:    bool = False,
    do_merge:      bool = False,
    push_encoded:  bool = False,
    ref_audio:    str = "",
    ref_text:     str = "",
    voice_name:   str = "default",
):
    if not skip_download:
        print(f"[1] Downloading {num_samples} samples from {hf_dataset} ...")
        download_data.remote(hf_dataset, num_samples)

    if not skip_filter:
        print("[2] Filtering dataset ...")
        filter_data.remote()

    if not skip_encode:
        print(f"[3] Encoding up to {max_encode} samples with NeuCodec ...")
        encode_data.remote(max_encode, hf_encoded_repo)
    elif push_encoded:
        print(f"[3b] Pushing existing encoded dataset to HF ...")
        push_encoded_data.remote(hf_encoded_repo)

    if not skip_train:
        print("[4] Training LoRA adapter ...")
        train.remote(hf_encoded_repo)

    if do_merge:
        print("[5] Merging LoRA into base model ...")
        merge_lora.remote()

    if ref_audio and ref_text:
        print(f"[6] Creating voices.json for voice '{voice_name}' ...")
        create_voices.remote(ref_audio, ref_text, voice_name)

    print("Pipeline complete.")
