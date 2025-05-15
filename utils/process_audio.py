import requests
from openai import OpenAI
from logs import logger
from typing import Union 
from gtts import gTTS
import pendulum

from typing import Union, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(".env"))

VERSION = os.getenv("VERSION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
API_URL = os.getenv("API_URL")
WHATSAPP_API_URL = f"{API_URL}/{VERSION}"
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")


client = OpenAI(api_key=OPENAI_API_KEY)


def download_audio(audio_id: str) -> Union[str, None]:
    """
    Télécharge un fichier audio depuis l'API WhatsApp et l'enregistre localement.

    Args:
        audio_id (str): L'identifiant unique du fichier audio à récupérer.

    Returns:
        str | None: Le chemin du fichier audio téléchargé en cas de succès, sinon None.
    
    Raises:
        requests.exceptions.RequestException: En cas d'erreur réseau lors de la requête.
    
    Remarque:
        - Cette fonction suppose que `WHATSAPP_API_URL` et `ACCESS_TOKEN` sont correctement configurés.
        - Le fichier audio est sauvegardé au format `.ogg` dans le dossier `./data/audio/`.
    """
    
    # Construire l'URL de l'API pour récupérer l'audio
    url = f"{WHATSAPP_API_URL}/{audio_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        # Récupération de l'URL du fichier audio
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lève une exception si le statut HTTP est une erreur

        audio_url = response.json().get("url")
        if not audio_url:
            logger.error(f"URL audio introuvable pour l'ID: {audio_id}")
            return None

        logger.info(f"🔗 Audio URL (download_audio): {audio_url}")
    except Exception as e:
        logger.exception(f"une errur lors de la recuperation du lien du fichier audio: {e}")

    def save_audio_file(audio_url:str=audio_url):
        # Télécharger le fichier audio
        audio_response = requests.get(audio_url, headers=headers)
        audio_response.raise_for_status()

        # Générer un timestamp unique
        dt = pendulum.now().format("YYYY_MM_DD_HH_mm_ss")
        audio_path = f"./data/audio/{audio_id}_{dt}.ogg"

        # Sauvegarder le fichier audio localement
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)

        logger.success(f"✅ Audio téléchargé avec succès: {audio_path}")
        return audio_path
    try:
        logger.info("premiere tentative de save l'audio")
        audio_path = save_audio_file()
        return audio_path
    except requests.exceptions.ReadTimeout:
        try:
            logger.info("deuxieme tentative de save l'audio")
            logger.warning(f"Probleme de connection tentative a nouveau de telecharger l'audio")
            audio_path = save_audio_file()
            return audio_path
        except requests.exceptions.ReadTimeout:
            logger.info("Troisième tentative de save l'audio")
            logger.warning(f"Probleme de connection tentative a nouveau de telecharger l'audio")
            audio_path = save_audio_file()
            return audio_path
        except requests.exceptions.RequestException as e:
            logger.exception(f"❌ Erreur lors du téléchargement de l'audio {audio_id}: {e}")
            return None
    except requests.exceptions.RequestException as e:
        logger.exception(f"❌ Erreur lors du téléchargement de l'audio {audio_id}: {e}")
        return None

    

def send_audio_message(to: str, audio_id: str) -> Optional[requests.Response]:
    """
    Envoie un message audio via l'API WhatsApp Business.

    Args:
        to (str): Numéro de téléphone du destinataire (format international).
        audio_id (str): L'identifiant du fichier audio stocké sur WhatsApp.
        phone_number_id (str): L'ID du numéro de téléphone WhatsApp Business utilisé pour envoyer le message. (obsolte)

    Returns:
        Optional[requests.Response]: L'objet `Response` en cas de succès, sinon `None`.

    Raises:
        requests.exceptions.RequestException: En cas d'échec de la requête HTTP.

    Exemple:
        >>> send_audio_message("+123456789", "AUDIO_ID")
    """

    logger.info(f"📤 Tentative d'envoi du message audio avec ID: {audio_id} à {to}")

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "audio",
        "audio": {"id": audio_id}
    }

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    url = f"{WHATSAPP_API_URL}/{PHONE_NUMBER_ID}/messages"

    try:
        response = requests.post(url=url, json=payload, headers=headers)
        response.raise_for_status()  # Lève une exception en cas d'erreur HTTP

        logger.success(f"✅ Message audio envoyé avec succès ! Statut: {response.status_code}")
        return response

    except requests.exceptions.RequestException as e:
        logger.exception(f"❌ Échec de l'envoi du message audio (ID: {audio_id}) : {e}")
        return None


def upload_audio_file(path: str) -> Optional[str]:
    """
    Télécharge un fichier audio sur l'API WhatsApp et renvoie son ID.

    Args:
        path (str): Chemin local du fichier audio à uploader.
        phone_number_id (str): L'ID du numéro WhatsApp Business utilisé. (obsolete)

    Returns:
        Optional[str]: L'identifiant du fichier média sur WhatsApp en cas de succès, sinon `None`.

    Raises:
        requests.exceptions.RequestException: En cas d'erreur de connexion ou de requête.

    Exemple:
        >>> media_id = upload_audio_file("./audio.mp3")
        >>> print(media_id)
    """

    logger.info(f"📤 Tentative d'upload du fichier audio: {path}")

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }

    try:
        with open(path, "rb") as audio_file:
            files = {
                "file": (path.split("/")[-1], audio_file, "audio/mpeg")
            }

            data = {"messaging_product": "whatsapp"}

            response = requests.post(
                url=f"{WHATSAPP_API_URL}/{PHONE_NUMBER_ID}/media",
                headers=headers,
                files=files,
                data=data,
            )

        # Vérifie si la requête a réussi
        response.raise_for_status()
        media_id = response.json().get("id")

        if media_id:
            logger.success(f"✅ Audio uploadé avec succès ! ID: {media_id}")
            return media_id
        else:
            logger.error(f"❌ Réponse invalide : ID du média manquant. {response.json()}")
            return None

    except requests.exceptions.RequestException as e:
        logger.exception(f"❌ Échec de l'upload de l'audio ({path}) : {e}")
        return None


    
def transcript_audio(path: str) -> Optional[str]:
    """
    Transcrit un fichier audio en texte en utilisant l'API de transcription.

    Args:
        path (str): Chemin du fichier audio à transcrire.

    Returns:
        Optional[str]: Le texte transcrit en cas de succès, sinon `None`.

    Raises:
        Exception: En cas d'échec de la transcription.

    Exemple:
        >>> texte = transcript_audio("./audio.ogg")
        >>> print(texte)
    """

    logger.info(f"🎤 Tentative de transcription du fichier audio : {path}")

    try:
        with open(path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )

        if transcript and hasattr(transcript, "text"):
            logger.success(f"✅ Transcription réussie : {transcript.text[:100]}")
            return transcript.text
        else:
            logger.error(f"❌ Échec de la transcription : réponse invalide. {transcript}")
            return None

    except Exception as e:
        logger.exception(f"❌ Une erreur s'est produite lors de la transcription : {e}")
        return None


def text_to_speech_with_openai(text: str, path: str) -> Optional[Tuple[str, object]]:
    """
    Convertit un texte en parole et enregistre le fichier audio généré.

    Args:
        text (str): Le texte à convertir en parole.
        path (str): Le chemin où sauvegarder le fichier audio (ex: "output.mp3").

    Returns:
        Optional[Tuple[str, object]]: 
            - Le chemin du fichier audio en cas de succès.
            - L'objet réponse de l'API OpenAI.
            - `None` en cas d'échec.

    Raises:
        Exception: En cas d'erreur pendant la génération du fichier audio.

    Exemple:
        >>> result = text_to_speech_with_openai("Bonjour", "speech.mp3")
        >>> print(result)
    """

    logger.info(f"🗣️ Tentative de conversion texte en audio. Fichier cible : {path}")

    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Prononce bien les mots de façon audible et naturelle sans accent.",
        )

        response.stream_to_file(path)  # Sauvegarde du fichier audio
        logger.success(f"✅ Fichier audio généré avec succès : {path}")

        # Affichage des détails de la réponse (facultatif)
        logger.info(f"🔊 Réponse Text-to-Speech : \tContenu -->> {response}")

        return path, response

    except Exception as e:
        logger.exception(f"❌ Erreur lors de la synthèse vocale : {e}")
        return None


def text_to_speech_with_gtts(text: str, filename: str, lang: str = "fr", tld: str = "fr") -> Optional[str]:
    """
    Convertit un texte en parole et enregistre le fichier audio généré à l'aide de gTTS.

    Args:
        text (str): Le texte à convertir en parole.
        filename (str): Le chemin du fichier de sortie (ex: "output.mp3").
        lang (str, optional): La langue de la voix (ex: "fr" pour français). Par défaut, "fr".
        tld (str, optional): Domaine de premier niveau pour l'accent (ex: "fr", "com", "co.uk"). Par défaut, "fr".

    Returns:
        Optional[str]: Le chemin du fichier audio en cas de succès, sinon `None`.

    Raises:
        Exception: En cas d'échec pendant la génération de l'audio.

    Exemple:
        >>> text_to_speech_with_gtts("Bonjour", "speech.mp3")
    """

    try:
        # Nettoyage du texte pour éviter les erreurs
        text_format = text.replace('"', "").replace("\n", " ").strip()

        # Création et sauvegarde du fichier audio
        tts = gTTS(text=text_format, lang=lang, tld=tld)
        tts.save(filename)

        logger.success(f"✅ Fichier audio généré avec succès : {filename}")
        return filename

    except Exception as e:
        logger.exception(f"❌ Erreur lors de la synthèse vocale avec gTTS : {e}")
        return None