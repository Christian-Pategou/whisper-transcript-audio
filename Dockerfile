# Utiliser une image CUDA officielle pour s'assurer de la compatibilité
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Installer Python 3.11, pip et cuDNN 9 pour CUDA 12
RUN apt-get update && apt-get install -y libcudnn9 libcudnn9-dev python3.11 python3-pip ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Désinstaller les paquets cuDNN/CUBLAS de pip (on utilise ceux de apt-get)
RUN python3 -m pip uninstall -y nvidia-cudnn-cu12 nvidia-cublas-cu12

# Vérifier si libcudnn est bien installé
RUN ls -l /usr/lib/x86_64-linux-gnu/libcudnn*

# Définir le chemin des bibliothèques
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

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
