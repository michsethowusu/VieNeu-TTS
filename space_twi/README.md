---
title: VieNeu-TTS Twi
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: docker
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# VieNeu-TTS — Asante Twi

Text-to-speech for Asante Twi, fine-tuned from [VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B) using LoRA on the [Asante Twi Bible Speech](https://huggingface.co/datasets/ghananlpcommunity/asante-twi-bible-speech-text) dataset.

## Inference modes

| Mode | Speed | Hardware |
|---|---|---|
| CPU (Standard) | ~1-2 min/sentence | Any CPU |
| CPU Fast (GGUF Q4) | ~10-30s/sentence | Any CPU |
| GPU | ~3-5s/sentence | CUDA GPU |
