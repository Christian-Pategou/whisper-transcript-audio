import noisereduce as nr
import librosa
import soundfile as sf

def reduce_bruit(path:str):
    # Charger l'audio
    data, rate = librosa.load(path=path, sr=None)
    # Réduction du bruit
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # Sauvegarder l'audio nettoyé
    sf.write(f"{path}_reduit.wav", reduced_noise, rate)