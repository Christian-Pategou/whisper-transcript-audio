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

ENV GROQ_API_KEY=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 GROQ_API_KEY_2=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 GROQ_MODEL_NAME=llama-3.3-70b-versatile GROQ_MODEL_NAME_2=llama-3.3-70b-versatile GOOGLE_API_KEY=AIzaSyBnSTLVHR4CAkD-h1BitWq5sxKSn4PSE5g
# Copier tout le code source dans le conteneur
COPY . .

# DÃ©marrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
