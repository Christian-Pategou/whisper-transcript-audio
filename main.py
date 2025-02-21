from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
from pydub import AudioSegment
import io

app = FastAPI()

# Load the Whisper model
model_size = "large-v3"  # You can choose other sizes like "base", "small", etc.
model = WhisperModel(model_size, device="cpu", compute_type="int8")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read the uploaded audio file
    audio = AudioSegment.from_file(io.BytesIO(await file.read()))

    # Export the audio to a format compatible with Whisper
    audio.export("temp_audio.wav", format="wav")

    # Transcribe the audio
    segments, info = model.transcribe("temp_audio.wav", beam_size=5)

    # Prepare the transcription result
    transcription = " ".join([segment.text for segment in segments])

    return {"transcription": transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)