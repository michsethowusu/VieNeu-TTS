FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy space files
COPY requirements.txt .

# Install llama-cpp-python pre-built wheel for Python 3.11 first,
# then the rest of the requirements
RUN pip install --no-cache-dir \
    --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/ \
    llama-cpp-python==0.3.16

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
