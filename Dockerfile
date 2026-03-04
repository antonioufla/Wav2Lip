# Wav2Lip - RunPod Serverless
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/workspace/results \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod==1.6.2 \
    librosa==0.9.2 \
    numpy==1.23.5 \
    opencv-contrib-python==4.8.0.76 \
    opencv-python==4.8.0.76 \
    tqdm \
    numba \
    requests \
    huggingface_hub \
    ffmpeg-python \
    batch-face

# Baixar pesos do modelo
RUN mkdir -p checkpoints && \
    huggingface-cli download Rudrabha/Wav2Lip \
      --local-dir checkpoints \
      --include "*.pth"

CMD ["python", "-u", "handler.py"]
