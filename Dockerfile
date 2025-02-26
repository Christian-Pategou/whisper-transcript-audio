# Utiliser une image CUDA officielle pour s'assurer de la compatibilité
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Installer Python 3.11 et pip
RUN apt-get update && apt-get install -y python3.11 python3-pip ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer cuDNN et cuBLAS via pip
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

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
