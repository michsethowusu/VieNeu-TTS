"""
Test the VieNeu-TTS library's GGUF inference path (what the Space now uses),
to confirm it generates real audio before rebuilding the Space.

  modal run finetune/test_gguf.py
"""
import modal

APP_NAME = "test-gguf-load"

# Mirror the Space environment exactly: vieneu --no-deps + stub + wheel + deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "espeak-ng", "libsndfile1", "git")
    .pip_install(
        "https://huggingface.co/michsethowusu/VieNeu-TTS-Twi/resolve/main/wheels/llama_cpp_python-0.3.25-py3-none-linux_x86_64.whl",
        "huggingface_hub", "phonemizer", "numpy", "onnxruntime", "soundfile",
    )
    .run_commands(
        "pip install --no-deps git+https://github.com/michsethowusu/VieNeu-TTS.git",
        # sea_g2p stub (Vietnamese G2P unused for Twi)
        "mkdir -p /usr/local/lib/python3.11/site-packages/sea_g2p",
        "printf 'class SEAPipeline:\\n    def __init__(self,*a,**k): pass\\n    def run(self,t): return t\\nclass G2P:\\n    def __init__(self,*a,**k): pass\\n    def phonemize_batch(self,t,**k): return list(t)\\nclass Normalizer:\\n    def normalize(self,t): return t\\n' > /usr/local/lib/python3.11/site-packages/sea_g2p/__init__.py",
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(cpu=4, memory=8192, timeout=900)
def test_infer(gguf_file: str = "VieNeu-TTS-Twi-Q4_K_M.gguf"):
    from vieneu import Vieneu

    tts = Vieneu(
        mode="standard",
        backbone_repo="michsethowusu/VieNeu-TTS-Twi",
        gguf_filename=gguf_file,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder-int8",
        codec_device="cpu",
        lang="twi",
        emotion="none",
    )

    voices = tts.list_preset_voices()
    print(f"Voices: {voices}")
    vname = tts._default_voice or voices[0][1]
    voice = tts.get_preset_voice(vname)

    wav = tts.infer("Meda wo ase paa. Onyame nhyira wo.", voice=voice,
                    temperature=0.4, apply_watermark=False)

    result = {
        "voice": vname,
        "wav_samples": int(len(wav)),
        "duration_s": round(len(wav) / tts.sample_rate, 2),
    }
    print("\n" + "=" * 50)
    print("RESULT:", result)
    print("=" * 50)
    return result


@app.local_entrypoint()
def main(gguf_file: str = "VieNeu-TTS-Twi-Q4_K_M.gguf"):
    result = test_infer.remote(gguf_file)
    print("\n>>> FINAL RESULT <<<")
    print(result)
