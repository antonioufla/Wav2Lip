# Wav2Lip - RunPod Serverless
# Updated: 2026-03-04 22:00 - Force rebuild
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

# Baixar pesos do modelo Wav2Lip via huggingface_hub (wget falha com exit 6 nos redirects do HF)
RUN mkdir -p checkpoints && \
    python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
src = hf_hub_download(repo_id='Rudrabha/Wav2Lip', filename='checkpoints/wav2lip_gan.pth'); \
shutil.copy(src, 'checkpoints/wav2lip_gan.pth'); \
print('Downloaded wav2lip_gan.pth')" && \
    ls -lh checkpoints/

# Baixar modelo de detecção de face (s3fd) necessário em runtime
RUN mkdir -p face_detection/detection/sfd && \
    wget -q --show-progress --no-check-certificate \
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" \
        -O face_detection/detection/sfd/s3fd.pth && \
    ls -lh face_detection/detection/sfd/

CMD ["python", "-u", "handler.py"]
