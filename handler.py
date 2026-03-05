import runpod
import os
import time
import requests
import subprocess
import uuid
import base64
from pathlib import Path

WORKSPACE = Path("/workspace")
OUTPUT_DIR = WORKSPACE / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = str(WORKSPACE / "checkpoints" / "wav2lip_gan.pth")
S3FD_PATH = WORKSPACE / "face_detection" / "detection" / "sfd" / "s3fd.pth"


def download_weights():
    """Download model weights at cold start if not present."""
    # Wav2Lip GAN checkpoint via mirror público (sem autenticação)
    cp = Path(CHECKPOINT_PATH)
    if not cp.exists():
        cp.parent.mkdir(parents=True, exist_ok=True)
        print("[Wav2Lip] Baixando wav2lip_gan.pth...")
        url = "https://huggingface.co/justinjohn0306/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        with open(str(cp), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[Wav2Lip] wav2lip_gan.pth salvo em {cp}")
    else:
        print(f"[Wav2Lip] checkpoint já existe: {cp}")

    # s3fd face detection model
    if not S3FD_PATH.exists():
        S3FD_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("[Wav2Lip] Baixando s3fd.pth...")
        url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        with open(str(S3FD_PATH), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"[Wav2Lip] s3fd.pth salvo em {S3FD_PATH}")
    else:
        print(f"[Wav2Lip] s3fd já existe: {S3FD_PATH}")


download_weights()


def download_file(url: str, dest: Path) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
    return dest


def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")

    if not image_url:
        return {"error": "image_url é obrigatório"}
    if not audio_url:
        return {"error": "audio_url é obrigatório"}

    job_id = job.get("id", str(uuid.uuid4()))
    tmp_dir = OUTPUT_DIR / job_id
    tmp_dir.mkdir(exist_ok=True)

    try:
        print(f"[Wav2Lip] Baixando imagem: {image_url}")
        image_path = download_file(image_url, tmp_dir / "input.png")

        print(f"[Wav2Lip] Baixando áudio: {audio_url}")
        audio_path = download_file(audio_url, tmp_dir / "input.wav")

        output_path = str(tmp_dir / "result.mp4")

        print(f"[Wav2Lip] Iniciando inferência...")
        start_time = time.time()

        cmd = [
            "python3", "inference.py",
            "--checkpoint_path", CHECKPOINT_PATH,
            "--face", str(image_path),
            "--audio", str(audio_path),
            "--outfile", output_path,
            "--static", "True",
            "--fps", "25",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE))

        if proc.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{proc.stderr}")

        elapsed = round(time.time() - start_time, 2)
        print(f"[Wav2Lip] Concluído em {elapsed}s")

        if not Path(output_path).exists():
            raise RuntimeError(f"Nenhum vídeo gerado em {output_path}")

        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "model": "Wav2Lip",
            "execution_time_seconds": elapsed,
            "video_base64": video_b64,
        }

    except Exception as e:
        return {"error": str(e), "model": "Wav2Lip"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
