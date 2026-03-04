# Wav2Lip - RunPod Serverless
# Updated: 2026-03-04 12:12 - Fixed dependencies
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/workspace/results \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev wget curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

# Instalar dependências compatíveis
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod==1.6.2 \
    librosa==0.9.2 \
    numpy==1.23.5 \
    scipy==1.10.1 \
    opencv-python==4.8.0.76 \
    tqdm \
    numba==0.56.4 \
    requests \
    huggingface_hub \
    ffmpeg-python \
    face-alignment \
    imageio==2.19.3 \
    imageio-ffmpeg

# Baixar pesos do modelo Wav2Lip (wav2lip_gan.pth)
RUN mkdir -p checkpoints && \
    python -c "import requests; url='https://huggingface.co/Rudrabha/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth'; r=requests.get(url, timeout=120); open('checkpoints/wav2lip_gan.pth','wb').write(r.content) if r.status_code==200 else exit(1)" && \
    ls -lh checkpoints/

CMD ["python", "-u", "handler.py"]
