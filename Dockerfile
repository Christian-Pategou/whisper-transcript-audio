FROM python:3.10-slim-bullseye

EXPOSE 8080

# Combiner les installations dans une seule commande
RUN apt-get update && \
    apt-get install --no-install-recommends -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

# Installer les packages dans une seule commande et sans cache
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt && \
    pip install --no-cache-dir transformers xformer && \
    pip install --no-cache-dir uvicorn requests && \
    pip install --no-cache-dir fastapi[all]

# DÃ©finir le rÃ©pertoire de travail
WORKDIR /app

ENV GOOGLE_API_KEY=AIzaSyBnSTLVHR4CAkD-h1BitWq5sxKSn4PSE5g HF_TOKEN=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
# Copier tout le code source dans le conteneur
COPY . .

# DÃ©marrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
