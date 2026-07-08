# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# ffmpeg/ffprobe are the only heavy system deps we need. No CUDA/ROCm base
# image, no local model weights baked in -- Track 2 compute is the
# Fireworks AI API, so the container stays small and starts fast.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PROVIDER=fireworks \
    MAX_FRAMES=16 \
    MODEL_TIMEOUT_SECONDS=120

# Contract: read /input/tasks.json, write /output/results.json
VOLUME ["/input", "/output"]

ENTRYPOINT ["python", "src/main.py"]
