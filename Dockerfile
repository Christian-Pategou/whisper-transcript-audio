# Image de base optimisée avec CUDA
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copier les fichiers du projet
COPY main.py ./

# Installation des dépendances Python
RUN pip3 install --no-cache-dir fastapi uvicorn \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    pydub numpy faster-whisper

# Exposition du port de l'API
EXPOSE 8000

# Lancer l'application avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
