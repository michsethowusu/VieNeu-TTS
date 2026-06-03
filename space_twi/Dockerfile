FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built llama-cpp-python CPU wheel — no compilation needed
RUN pip install --no-cache-dir \
    https://huggingface.co/michsethowusu/VieNeu-TTS-Twi/resolve/main/wheels/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/michsethowusu/VieNeu-TTS.git

COPY sea_g2p_stub /usr/local/lib/python3.11/site-packages/sea_g2p

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
