# Utiliser une image CUDA officielle pour s'assurer de la compatibilité
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Installer Python 3.11, pip et cuDNN pour CUDA 12
RUN apt-get update && apt-get install -y libcudnn8 libcudnn8-dev python3.11 python3-pip ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer cuDNN et cuBLAS via pip
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Vérifier l'installation de cuDNN
RUN python3 -c "import torch; print(torch.backends.cudnn.version())"

# Définir le chemin des bibliothèques
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Définir le dossier de travail
WORKDIR /app

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

# Exposer le port
EXPOSE 8000

# Lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
