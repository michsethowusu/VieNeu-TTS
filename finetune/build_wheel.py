"""
Build a llama-cpp-python CPU wheel on Modal and upload to HuggingFace Hub.

Usage:
  modal run finetune/build_wheel.py
"""

import os
import modal

APP_NAME = "build-llama-cpp-wheel"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("cmake", "build-essential", "git")
    .pip_install("wheel", "huggingface_hub")
)

app = modal.App(APP_NAME, image=image)

@app.function(
    cpu=8,
    memory=16384,
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build_and_upload(hf_repo: str, version: str = "0.3.16") -> str:
    import subprocess
    import glob
    import os
    from huggingface_hub import HfApi

    wheel_dir = "/tmp/wheels"
    os.makedirs(wheel_dir, exist_ok=True)

    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF"

    print(f"Building llama-cpp-python=={version} wheel (CPU only) ...")
    subprocess.run(
        [
            "pip", "wheel",
            "--no-deps",
            "--wheel-dir", wheel_dir,
            f"llama-cpp-python=={version}",
        ],
        env=env,
        check=True,
    )

    wheels = glob.glob(f"{wheel_dir}/*.whl")
    if not wheels:
        raise RuntimeError("No wheel file found after build.")
    wheel_path = wheels[0]
    wheel_name = os.path.basename(wheel_path)
    print(f"Built: {wheel_name}")

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=wheel_path,
        path_in_repo=f"wheels/{wheel_name}",
        repo_id=hf_repo,
        repo_type="model",
        commit_message=f"Add pre-built llama-cpp-python {version} CPU wheel for Linux x86_64 Python 3.11",
    )

    url = f"https://huggingface.co/{hf_repo}/resolve/main/wheels/{wheel_name}"
    print(f"Uploaded → {url}")
    return url


@app.local_entrypoint()
def main(
    hf_repo: str = "michsethowusu/VieNeu-TTS-Twi",
    version: str = "0.3.16",
):
    print(f"[local] Building llama-cpp-python=={version} on Modal and uploading to {hf_repo} ...")
    url = build_and_upload.remote(hf_repo, version)
    print(f"\n[local] Wheel URL: {url}")
    print(f"\n[local] Add to Dockerfile:\n  RUN pip install --no-cache-dir {url}")
