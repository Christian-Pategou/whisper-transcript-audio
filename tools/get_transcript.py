import whisper
from whisper.model import Whisper
import rich

model = whisper.load_model("small")
# print("model successfully loaded")

def get_transcript(audio:str="./../audio_reduit.wav", model:Whisper=model) -> str:
    try:
        prompt = "Ce fichier audio contient une transcription en français sur le sujet de la santé et de la médecine. \
            Donc il y'a beaucoup de mots en rapport le lexique médical, le corps humain, les symptômes, le nom des médicaments."
        transcrib = model.transcribe(audio=audio, language="fr", initial_prompt=prompt)
        text = transcrib["text"]
        return text
    except	Exception as e:
        rich.print(f"\n\nerror occured \t\t{e}")


