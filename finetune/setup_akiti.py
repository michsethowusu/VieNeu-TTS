"""
One-time setup for the Akiti-TTS demo:
  1. Create michsethowusu/Akiti-TTS model repo
  2. Copy GGUF + tokenizer + wheel from VieNeu-TTS-Twi
  3. Encode reference_audio/*.wav -> voices.json, upload

  modal run finetune/setup_akiti.py
"""
import os
import modal

APP_NAME = "akiti-setup"
SRC_REPO = "michsethowusu/VieNeu-TTS-Twi"
DST_REPO = "michsethowusu/Akiti-TTS"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch", "torchaudio", "neucodec>=0.0.4", "librosa>=0.10.0",
                 "soundfile", "huggingface_hub>=0.23.0",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A10G", timeout=1800, memory=16384,
              secrets=[modal.Secret.from_name("huggingface-secret")])
def setup(audio_files: dict, metadata_lines: list):
    import io, json
    import librosa, torch
    from neucodec import NeuCodec
    from huggingface_hub import HfApi, snapshot_download

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    # 1) Create destination repo
    api.create_repo(DST_REPO, repo_type="model", exist_ok=True, private=False)
    print(f"Created {DST_REPO}")

    # 2) Copy artifacts from source repo (GGUF, tokenizer, wheel, safetensors)
    print(f"Copying artifacts from {SRC_REPO} ...")
    local = snapshot_download(repo_id=SRC_REPO, repo_type="model", token=hf_token)
    api.upload_folder(folder_path=local, repo_id=DST_REPO, repo_type="model",
                      commit_message="Copy model artifacts from VieNeu-TTS-Twi")
    print("Artifacts copied.")

    # 3) Encode reference audios -> voices.json
    print("Encoding reference audios with NeuCodec ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    codec.eval()

    voices = {
        "meta": {"spec": "vieneu.voice.presets", "spec_version": "1.0", "engine": "Akiti-TTS"},
        "default_voice": None,
        "presets": {},
    }
    for line in metadata_lines:
        if "|" not in line:
            continue
        fname, text = line.split("|", 1)
        fname, text = fname.strip(), text.strip()
        if fname not in audio_files or not text:
            continue
        wav, _ = librosa.load(io.BytesIO(audio_files[fname]), sr=16000, mono=True)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            codes = codec.encode_code(wav_t).squeeze(0).squeeze(0).cpu().numpy().flatten().tolist()
        codes = [int(c) for c in codes]
        name = os.path.splitext(fname)[0]   # "kofi.wav" -> "kofi"
        voices["presets"][name] = {"codes": codes, "text": text,
                                    "description": f"Akiti voice: {name}"}
        if voices["default_voice"] is None:
            voices["default_voice"] = name
        print(f"  {name}: {len(codes)} codes")

    tmp = "/tmp/voices.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)
    api.upload_file(path_or_fileobj=tmp, path_in_repo="voices.json",
                    repo_id=DST_REPO, repo_type="model",
                    commit_message="Add voices.json from reference audio")
    print(f"Uploaded voices.json with {len(voices['presets'])} voices.")
    return list(voices["presets"].keys())


@app.local_entrypoint()
def main():
    import pathlib
    ref_dir = pathlib.Path(__file__).parent / "reference_audio"
    meta = [l.strip() for l in (ref_dir / "metadata.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    audio = {}
    for line in meta:
        fname = line.split("|", 1)[0].strip()
        p = ref_dir / fname
        if p.exists():
            audio[fname] = p.read_bytes()
    print(f"[local] Sending {len(audio)} reference audios to Modal ...")
    voices = setup.remote(audio, meta)
    print(f"[local] Done. Voices: {voices}")
