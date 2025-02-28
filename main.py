from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydub import AudioSegment
import numpy as np
import torch
import io

app = FastAPI()

# Ajoute le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes HTTP
    allow_headers=["*"],   # Permet tous les en-têtes
)

# Vérifie si CUDA est disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilisation de l'appareil : {device}")

# Charge le modèle Whisper optimisé
model_size = "large-v3"
model = WhisperModel(model_size, device=device, compute_type="float16", num_workers=4)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Lire le fichier audio sans l'écrire sur le disque
    audio = AudioSegment.from_file(io.BytesIO(await file.read()))

    # Convertir en tableau numpy et normaliser
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    # Transcrire l'audio
    segments, info = model.transcribe(samples, beam_size=5)  # Beam size réduit pour plus de rapidité

    print(f"Langue détectée : {info.language} (Probabilité : {info.language_probability:.2%})")

    # Construire la transcription
    transcription = " ".join([segment.text for segment in segments])

    return {"transcription": transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
