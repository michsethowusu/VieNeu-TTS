---
title: Akiti TTS
emoji: 🪘
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# 🪘 Akiti TTS — Asante Twi Text-to-Speech

A lightweight Gradio frontend for Akiti TTS. Inference runs on-demand on Modal
(CPU, scale-to-zero), so the Space itself stays light and the model only runs
when someone makes a request.

Fine-tuned from [VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B).
Model & voices: [michsethowusu/Akiti-TTS](https://huggingface.co/michsethowusu/Akiti-TTS).

## Setup (Space secrets)

This Space calls a deployed Modal app. Add these as Space **secrets**:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Create them with `modal token new` (or from the Modal dashboard) for the
workspace where `akiti-tts` is deployed.
