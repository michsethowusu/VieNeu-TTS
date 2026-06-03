FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the full repo so apps/ and src/ are importable
RUN git clone --depth=1 https://github.com/michsethowusu/VieNeu-TTS.git /opt/vieneutts

ENV PYTHONPATH="/opt/vieneutts"

# Install pre-built CPU wheel — no compilation needed
RUN pip install --no-cache-dir \
    https://huggingface.co/michsethowusu/VieNeu-TTS-Twi/resolve/main/wheels/llama_cpp_python-0.3.16-cp311-cp311-linux_x86_64.whl

# Install remaining dependencies from the repo
RUN pip install --no-cache-dir -r /opt/vieneutts/requirements.txt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy space-specific files
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
