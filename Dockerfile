# Utiliser une image CUDA officielle
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Installer les dépendances (Python 3.11, cuDNN pour CUDA 12, et ffmpeg)
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip ffmpeg \
    libcudnn9-cuda-12 libcudnn9-dev-cuda-12 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Vérifier que cuDNN est bien installé
RUN ls -l /usr/lib/x86_64-linux-gnu/libcudnn*

# Définir le chemin des bibliothèques pour s'assurer que cuDNN est détecté
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Définir le dossier de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Installation séparée de PyTorch avec l'index CUDA
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copier le reste du code
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
