# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b9PO-lcGZX9pEkEwQmu8MfhSnjxKrALW?usp=sharing)
[![Hugging Face VieNeu-TTS-v2](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v2-blue)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2)
[![Hugging Face VieNeu-TTS](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-v1-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)

<img width="1087" height="710" alt="image" src="https://github.com/user-attachments/assets/5534b5db-f30b-4d27-8a35-80f1cf6e5d4d" />

**VieNeu-TTS-v2** là thế hệ tiếp theo của mô hình chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt chạy trên thiết bị, hỗ trợ **10.000+ giờ dữ liệu** huấn luyện song ngữ, **clone giọng nói tức thì**, và chế độ **Podcast/Hội thoại** chuyên dụng.

> [!IMPORTANT]
> **🚀 VieNeu-TTS-v2 đã ra mắt!**
> Kiến trúc song ngữ chất lượng cao (high-fidelity) hiện đã sẵn sàng với:
> - **10.000+ Giờ dữ liệu:** Độ tự nhiên vượt trội trong cả tiếng Anh và tiếng Việt.
> - **Chế độ Podcast & Đối thoại:** Hỗ trợ đa người nói với các sắc thái biểu cảm.
> - **Zero-shot Cloning:** Clone bất kỳ giọng nói nào chỉ trong 3-5 giây trên tất cả các biến thể v2.

## ✨ Tính năng nổi bật
- **Huấn luyện 10.000+ giờ**: Được huấn luyện trên tập dữ liệu Anh-Việt khổng lồ cho ngữ điệu giống hệt con người.
- **Song ngữ (En-Vi) Code-switching**: Chuyển đổi ngôn ngữ mượt mà ngay trong câu.
- **Chế độ Podcast & Hội thoại**: Hỗ trợ đối thoại đa người nói với khả năng tự động nhận diện nhân vật.
- **Clone giọng nói tức thì**: Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh mẫu.
- **Hiệu suất cực nhanh**: Được tối ưu hóa cho **GPU (LMDeploy)** và **CPU (GGUF/ONNX)**.
- **Sẵn sàng cho sản xuất**: Tạo âm thanh chất lượng cao 24 kHz, hoạt động hoàn toàn offline.

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd" />](https://github.com/user-attachments/assets/021f6671-2d7f-4635-91fb-88b2ab0ddbcd)

## 📌 Mục lục

1. [🦜 Cài đặt & Giao diện Web](#installation)
2. [📦 Sử dụng Python SDK](#sdk)
3. [🐳 Server Chất lượng cao (Standard Mode)](#docker-remote)
4. [🔬 Tổng quan mô hình](#backbones)
5. [🚀 Lộ trình phát triển](#roadmap)
6. [🤝 Hỗ trợ & Liên hệ](#support)
7. [📑 Trích dẫn](#citation)

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>

### Thiết lập với `uv` (Khuyến nghị)
`uv` là cách nhanh nhất để quản lý các phụ thuộc.
```bash
# Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt các phụ thuộc:**
   - **Lựa chọn 1: Tối giản (Turbo/CPU)** - Nhanh & Nhẹ
     > ⚠️ *Lưu ý: Chế độ này chỉ hỗ trợ **VieNeu-TTS-v2-Turbo (CPU)** — chạy được trên mọi máy không cần GPU, nhưng **chất lượng âm thanh thấp hơn** so với Standard VieNeu-TTS (đặc biệt với câu ngắn < 5 từ). Khuyến nghị dùng để thử nghiệm nhanh hoặc triển khai trên thiết bị yếu.*
     ```bash
     uv sync
     ```
   - **Lựa chọn 2: Đầy đủ (GPU/Standard)** - Chất lượng cao & Chế độ Podcast *(Dành cho người dùng GPU)*
     > 💡 *Lưu ý: Yêu cầu GPU NVIDIA hỗ trợ CUDA (phiên bản CUDA >= 12.8) hoặc Apple Silicon MPS. Cần cài đặt [NVIDIA Toolkit](https://developer.nvidia.com/cuda-downloads) để đạt tốc độ tối đa. Kích hoạt toàn bộ backbone **VieNeu-TTS-v2** để đạt chất lượng âm thanh tối đa và clone giọng nói độ trung thực cao.*
     ```bash
     uv sync --group gpu
     ```

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run vieneu-web
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`. 

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

SDK `vieneu` mặc định sử dụng **chế độ Standard** (0.3B GGUF + ONNX) khi dùng cục bộ, mang lại sự cân bằng hoàn hảo giữa chất lượng âm thanh cao và tốc độ xử lý thời gian thực trên cả CPU và GPU.

### Bắt đầu nhanh
```bash
# Cài đặt tối giản (Build llama-cpp từ nguồn - có thể mất chút thời gian)
pip install vieneu

# Tùy chọn: Dành cho người dùng Windows (CPU pre-built)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Tùy chọn: Dành cho người dùng macOS (ARM64/Apple Silicon - Kích hoạt Metal GPU)
pip install vieneu --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal/
```

```python
from vieneu import Vieneu

# Khởi tạo chế độ Standard (Mặc định - Chất lượng cao nhất)
# Chạy ngay lập tức trên CPU mà không cần PyTorch!
tts = Vieneu(emotion="natural") # emotion="natural" (giọng tự nhiên - mặc định) hoặc "storytelling" (giọng kể chuyện)

# 1. Tổng hợp đơn giản (sử dụng giọng Nữ miền Bắc mặc định 'Trúc Ly')
text = "Chào bạn. Tôi là VieNeu-TTS, tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, thậm chí clone giọng nói của bạn."
audio = tts.infer(text=text)

# Lưu thành file
tts.save(audio, "output_Trúc Ly.wav")
print("💾 Đã lưu file output_Trúc Ly.wav")

# 2. Sử dụng Giọng mẫu cụ thể (Preset Voice)
voices = tts.list_preset_voices()
for desc, voice_id in voices:
    print(f"Giọng: {desc} (ID: {voice_id})")

my_voice_id = voices[1][1] if len(voices) > 1 else voices[0][1] # Giọng Phạm Tuyên
voice_data = tts.get_preset_voice(my_voice_id)

audio_custom = tts.infer(text="Tôi đang nói bằng giọng của Bác sĩ Tuyên.", voice=voice_data)

# 3. Lưu thành file
tts.save(audio_custom, "output_Phạm Tuyên.wav")
print("💾 Đã lưu file output_Phạm Tuyên.wav")
```

### 🚀 Chế độ Turbo (Song ngữ & Tốc độ cực nhanh)
Sử dụng `mode="turbo"` để đạt tốc độ xử lý nhanh nhất, đặc biệt tối ưu cho việc đọc song ngữ Anh-Việt (code-switching) trong thời gian thực.

> [!WARNING]
> Chế độ Turbo có chất lượng âm thanh thấp hơn các chế độ khác và thường gặp lỗi (bị lặp hoặc mất chữ) đối với các câu quá ngắn.

```python
from vieneu import Vieneu

# Khởi tạo chế độ Turbo (v2-Turbo GGUF)
tts = Vieneu(mode="turbo")

# Turbo v2 hỗ trợ chuyển đổi Anh-Việt cực kỳ tự nhiên
text = "Hệ thống điện chủ yếu sử dụng alternating current because it is more efficient."
audio = tts.infer(text=text)

tts.save(audio, "turbo_output.wav")
```

### 🦜 Clone giọng nói Zero-shot (SDK) <a name="cloning"></a>
Clone bất kỳ giọng nói nào chỉ với **3-5 giây** âm thanh. 

> [!TIP]
> **Chế độ Turbo** được khuyến nghị cho việc clone giọng vì không yêu cầu văn bản mẫu (`ref_text`), trong khi **chế độ Standard** (mặc định) yêu cầu cung cấp `ref_text` để đạt độ chính xác cao hơn.

```python
from vieneu import Vieneu

# Sử dụng turbo mode để clone giọng dễ dàng (không cần ref_text)
tts = Vieneu(mode="turbo")

# 1. Trích xuất đặc trưng giọng nói (3-5 giây khuyến nghị)
my_voice = tts.encode_reference("examples/audio_ref/example.wav")

# 2. Tổng hợp với giọng đã clone
audio = tts.infer(
    text="Đây là giọng nói được clone trực tiếp bằng SDK của VieNeu-TTS.", 
    voice=my_voice
)

tts.save(audio, "cloned_voice.wav")
```

---

## 🐳 3. Server Chất lượng cao (Standard Mode) <a name="docker-remote"></a>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (được hỗ trợ bởi LMDeploy) chỉ bằng một câu lệnh duy nhất.

### 1. Chạy với Docker (Khuyến nghị)

**Yêu cầu**: Cần cài đặt [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) để hỗ trợ GPU.

**Khởi chạy Server với Đường hầm công khai (Không cần mở cổng modem):**
```bash
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```

*   **Mặc định**: Server sẽ tải model `VieNeu-TTS` (v1/v2) để đạt chất lượng tối đa.
*   **Tunneling**: Docker image tích hợp sẵn đường hầm `bore`. Kiểm tra container logs để tìm địa chỉ công khai của bạn (VD: `bore.pub:31631`).

### 2. Sử dụng SDK (Chế độ Remote)

Khi server đã chạy, bạn có thể kết nối từ bất kỳ đâu (Colab, Web App, v.v.) mà không cần tải các model nặng cục bộ:

```python
from vieneu import Vieneu
import os

# Cấu hình
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Hoặc URL từ bore tunnel
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS"

# Khởi tạo (Cực kỳ NHẸ - chỉ tải codec nhỏ cục bộ)
tts = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID)
os.makedirs("outputs", exist_ok=True)

# Liệt kê các giọng mẫu trên server
available_voices = tts.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Sử dụng giọng cụ thể
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = tts.get_preset_voice(my_voice_id)
    audio_spec = tts.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Đã lưu kết quả tại: outputs/remote_{my_voice_id}.wav")

# Tổng hợp chuẩn (dùng giọng mặc định)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = tts.infer(text=text_input)
tts.save(audio, "outputs/remote_output.wav")
print("💾 Đã lưu kết quả remote_output.wav")

# Clone giọng Zero-shot (Mã hóa âm thanh cục bộ, gửi code lên server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = tts.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    tts.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Đã lưu kết quả remote_cloned_output.wav")
```
*Chi tiết xem tại: [examples/main_remote.py](examples/main_remote.py)*

---

## 🔬 4. Tổng quan mô hình <a name="backbones"></a>

| Model | Định dạng | Thiết bị | Song ngữ | Tính năng | Tốc độ |
|---|---|---|---|---|---|
| **VieNeu-TTS-v2** | PyTorch | **GPU** | ✅ | **Podcast, En-Vi CS** | **Nhanh (LMDeploy)** |
| **VieNeu-v2-CPU** | GGUF/ONNX | **CPU/Edge** | ✅ | **Podcast, En-Vi CS** | **Rất nhanh** |
| **VieNeu-v2-Turbo** | GGUF/ONNX | **CPU/Edge** | ✅ | En-Vi mượt mà | **Cực nhanh** |
| **VieNeu-TTS (v1)** | PyTorch | GPU/CPU | ❌ | Ổn định (Chỉ Tiếng Việt) | Chuẩn |

> [!TIP]
> Sử dụng **Turbo v2** cho trợ lý AI, chatbot và các ứng dụng thời gian thực trên thiết bị yếu. Lưu ý: Có thể gặp vấn đề ổn định với các câu cực ngắn (< 5 từ).
> Sử dụng **GPU/Standard** (VieNeu-TTS v1/v2) để đạt chất lượng âm thanh tối đa và clone giọng độ trung thực cao.

---

## 🚀 5. Lộ trình phát triển <a name="roadmap"></a>

- [x] **VieNeu-TTS-v2**: Kiến trúc song ngữ chất lượng cao đầy đủ với **Chế độ Podcast** và **Clone giọng nói**.
- [x] **VieNeu-Codec**: Neural codec tối ưu cho tiếng Việt (ONNX).
- [x] **Turbo Voice Cloning**: Mang tính năng clone giọng nói tức thì lên engine Turbo siêu nhẹ.
- [ ] **Mobile SDK**: Hỗ trợ chính thức cho việc triển khai trên Android/iOS.

---

## 🤝 6. Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/pnnbao97)
- **Giấy phép:** Apache 2.0 (Sử dụng tự do).

---

## 📑 7. Trích dẫn <a name="citation"></a>

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS-v2: Advanced Vietnamese Text-to-Speech with Podcast and Code-Switching Support},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=pnnbao97/VieNeu-TTS&type=Date)](https://star-history.com/#pnnbao97/VieNeu-TTS&Date)

---

## 🤝 Người đóng góp

Cảm ơn tất cả những người tuyệt vời đã đóng góp cho dự án này!

<a href="https://github.com/pnnbao97/VieNeu-TTS/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pnnbao97/VieNeu-TTS" />
</a>

---

## 🙏 Lời cảm ơn

Dự án này sử dụng [neucodec](https://huggingface.co/neuphonic/neucodec) để giải mã âm thanh và [sea-g2p](https://github.com/pnnbao97/sea-g2p) để chuẩn hóa văn bản và phiên âm.

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
