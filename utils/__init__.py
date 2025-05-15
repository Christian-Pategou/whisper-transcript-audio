from logs import logger
from typing import Optional
from utils.process_audio import download_audio, transcript_audio


def process_audio_messages(audio_id: str) -> Optional[str]:
    """
    Traite un message audio WhatsApp : télécharge le fichier et le transcrit en texte.

    Args:
        audio_id (str): L'identifiant du fichier audio sur WhatsApp.

    Returns:
        Optional[str]: Le texte transcrit en cas de succès, sinon `None`.

    Raises:
        ValueError: Si le fichier audio n'a pas pu être téléchargé.
        RuntimeError: Si la transcription échoue.

    Exemple:
        >>> transcript = process_audio_messages("AUDIO_ID")
        >>> print(transcript if transcript else "Échec de la transcription")
    """

    logger.info(f"🎧 Début du traitement du message audio. ID: {audio_id}")

    # 1. Télécharger le fichier audio
    audio_path = download_audio(audio_id=audio_id)
    if not audio_path:
        logger.error("❌ Impossible de télécharger le fichier audio.")
        return None
        # raise ValueError("Le fichier audio n'a pas pu être téléchargé.")

    # logger.success(f"✅ Fichier audio téléchargé avec succès : {audio_path}")

    # 2. Transcrire l'audio
    text_audio = transcript_audio(path=audio_path)
    if isinstance(text_audio, str):
        # logger.success(f"✅ Transcription réussie : {text_audio[:50]}...")  # Affiche un extrait du texte
        return text_audio
    else:
        logger.error("❌ Échec de la transcription de l'audio.")
        raise RuntimeError("La transcription de l'audio a échoué.")

        


