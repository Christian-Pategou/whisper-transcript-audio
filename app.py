from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
import requests
import os, rich
from typing import Optional, Dict

from utils import process_audio_messages
from utils.process_audio import (
    send_audio_message,
    text_to_speech_with_gtts, 
    text_to_speech_with_openai,
    upload_audio_file,
)
from database.create_database import (
    get_or_create_user,
    get_or_create_conversation_id,
    save_message, 
    update_message_status,
)

from utils.format_messages import format_whatsapp_markdown

from logs import logger

app = FastAPI()

# Ajoute le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet uniquement les origines spÃ©cifiÃ©es
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les mÃ©thodes HTTP (GET, POST, etc.)
    allow_headers=["*"],   # Permet tous les en-tÃªtes
)

# --------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------

load_dotenv(find_dotenv(".env"))

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERSION = os.getenv("VERSION")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")

message = """
*Hello !* 👋  

Avant de commencer, je tiens à préciser que *_je ne suis pas un médecin_*.    

Je peux vous fournir des informations générales, mais *seul un professionnel de santé peut établir un diagnostic fiable.* 


➡️ Pense à consulter un professionnelle de santé en cas de besoin. 🏥  
"""

API_URL = os.getenv("API_URL")

URL = f"{API_URL}/{VERSION}/{PHONE_NUMBER_ID}/messages"

def collect_informations(body: dict) -> Optional[Dict[str, Optional[str]]]:
    """
    Extrait les informations pertinentes d'un message WhatsApp reçu via Webhook.

    Args:
        body (dict): Le dictionnaire contenant les données de l'événement WhatsApp.

    Returns:
        Optional[Dict[str, Optional[str]]]: Un dictionnaire avec les informations de l'utilisateur et du message,
        ou `None` si les données ne sont pas valides.

    Exemple:
        >>> data = collect_informations(webhook_body)
        >>> if data:
        >>>     print(data["user_phone_number"])
    """

    try:
        # Vérification des clés avant l'accès
        entry = body.get("entry", [])
        if not entry:
            logger.warning("❌ Aucune entrée trouvée dans le corps de la requête.")
            return None

        changes = entry[0].get("changes", [])
        if not changes:
            logger.warning("❌ Aucun changement détecté dans les entrées.")
            return None

        messages = changes[0].get("value", {})
        if "messages" not in messages:
            logger.info("ℹ️ Événement ignoré (pas de messages).")
            return None  # On ignore l'événement s'il ne contient pas de message

        # Extraction des informations de l'utilisateur
        contacts = messages.get("contacts", [{}])[0]
        user_phone_number = contacts.get("wa_id")
        user_profil_name = contacts.get("profile", {}).get("name", "Unknown")

        # Extraction des informations du message
        message_data = messages.get("messages", [{}])[0]
        user_conversation_id = message_data.get("id", "Unknown")
        user_conversation_timestamp = message_data.get("timestamp", "Unknown")
        user_messages_type = message_data.get("type", "Unknown")

        # Gestion des types de messages
        user_messages_content = None
        user_audio_id = None

        if user_messages_type == "audio":
            user_audio_id = message_data.get("audio", {}).get("id", "")
        elif user_messages_type == "text":
            user_messages_content = message_data.get("text", {}).get("body", "")

        # Résumé des informations collectées
        extracted_data = {
            "user_phone_number": user_phone_number,
            "user_profil_name": user_profil_name,
            "user_conversation_id": user_conversation_id,
            "user_conversation_timestamp": user_conversation_timestamp,
            "user_messages_type": user_messages_type,
            "user_messages_content": user_messages_content,
            "user_audio_id": user_audio_id,
        }

        logger.success(f"✅ Informations collectées avec succès.")
        return extracted_data

    except Exception as e:
        logger.exception(f"❌ Erreur lors de la collecte des informations : {e}")
        return None

def stream_chat(message: str, id: str, name: str) -> Optional[str]:
    """
    Envoie un message à un agent de chat via une API locale et récupère la réponse.

    Args:
        message (str): La question à poser à l'agent.
        id (str): Identifiant de l'utilisateur ou de la session.
        name (str): Nom de l'utilisateur.

    Returns:
        Optional[str]: La réponse de l'agent en cas de succès, sinon un message d'erreur.

    Raises:
        requests.exceptions.RequestException: En cas de problème de connexion à l'API.

    Exemple:
        >>> response = stream_chat("Bonjour, comment allez-vous ?", "12345", "Alice")
        >>> print(response)
    """

    url = AGENT_ENDPOINT

    try:
        logger.info(f"📨 Envoi du message à l'agent.")

        response = requests.post(url, json={"question": message, "id": id, "name": name})
        response.raise_for_status()  # Vérifie les erreurs HTTP

        logger.success("✅ Réponse reçue avec succès.")
        return response.text  # On retourne le texte de la réponse

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur de connexion avec l'API : {e}")
        return "Erreur lors de la consultation de l'agent."

    except Exception as e:
        logger.exception(f"❌ Une erreur inattendue est survenue : {e}")
        return "Erreur lors de l'appel à l'agent."


def send_whatsapp_message(call: str, data: Dict[str, str]) -> Optional[requests.Response]:
    """
    Envoie un message WhatsApp formaté à un utilisateur spécifique.

    Args:
        call (str): Le message brut à envoyer.
        data (Dict[str, str]): Dictionnaire contenant les informations de l'utilisateur (ex: numéro de téléphone).

    Returns:
        Optional[requests.Response]: La réponse de l'API si l'envoi réussit, sinon `None`.

    Raises:
        requests.exceptions.RequestException: En cas d'erreur de connexion avec l'API.

    Exemple:
        >>> response = send_whatsapp_message("Bonjour !", {"user_phone_number": "+123456789"})
        >>> print(response.status_code if response else "Échec de l'envoi")
    """

    phone_number = data.get("user_phone_number")
    if not phone_number:
        logger.error("❌ Numéro de téléphone manquant dans les données.")
        return None

    try:
        formatted_message = format_whatsapp_markdown(call).replace("\n\n", " " * 20).replace("\n", " " * 10)

        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": phone_number,
            "type": "text",
            "text": {"preview_url": False, "body": formatted_message.replace('"', "")}
        }

        logger.info(f"📨 Envoi d'un message à {phone_number}")
        response = requests.post(url=f"{API_URL}/{VERSION}/{PHONE_NUMBER_ID}/messages", headers=headers, json=payload)
        response.raise_for_status()  # Vérifie si la requête a échoué

        logger.success(f"✅ Message envoyé avec succès à {phone_number} (Statut: {response.status_code})")
        return response

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors de l'envoi du message WhatsApp : {e}")
        return None

@app.get("/webhook")
async def verify_webhook(mode: str = None, challenge: int = None, token: str = None):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("WEBHOOK_VERIFIED")
        return challenge, 200
    else:
        print("VERIFICATION_FAILED")
        return {"error": "Invalid token"}
    

# Set of processed messages to avoid duplicates
processed_messages = set()

@app.post("/webhook")
async def receive_message(request: Request):
    """
    Webhook endpoint pour recevoir et traiter les messages entrants.
    
    Args:
        request (Request): Requête HTTP contenant les informations du message entrant.
        
    Returns:
        dict: Message indiquant si le traitement a réussi ou non.
    """

    body = await request.json()
    logger.info(f"Corps du message reçu -->> {body}")
    data = collect_informations(body)

    if not data:
        return {"message": "Événement ignoré"}  # On ignore les événements non pertinents

    # Vérifier si c'est un message utilisateur
    if "messages" not in body["entry"][0]["changes"][0]["value"]:
        return {"message": "Non traité - pas un message utilisateur"}

    # Traitement des messages texte
    if data["user_messages_type"] == "text":
        return await handle_text_message(data)

    # Traitement des messages audio
    elif data["user_messages_type"] == "audio":
        return await handle_audio_message(data)

    return {"message": "Type de message non pris en charge"}  # Cas par défaut

async def handle_text_message(data: Dict[str, str]) -> dict:
    """
    Gère le traitement des messages texte entrants.
    
    Args:
        data (dict): Dictionnaire contenant les informations du message utilisateur.

    Returns:
        dict: Message indiquant si le traitement a réussi ou non.
    """
    user_phone = data["user_phone_number"]
    user_name = data.get("user_profil_name", "Unknown")

    # 1️⃣ Vérifier/créer l'utilisateur
    user_id = get_or_create_user(phone=user_phone, name=user_name)

    # 2️⃣ Récupérer un ID de conversation valide
    conversation_id, new_conversation = get_or_create_conversation_id(user_id)
    
    # Si c'est une nouvelle conversation, envoyer un message d'introduction
    if new_conversation:
        send_whatsapp_message(message, data)

    # 3️⃣ Sauvegarder le message utilisateur
    save_message(conversation_id, "user", data["user_messages_content"])

    # 4️⃣ Appeler l'agent pour obtenir une réponse
    agent_response = stream_chat(data["user_messages_content"], conversation_id, data["user_profil_name"])

    # Sauvegarder la réponse de l'agent
    save_message(conversation_id, "bot", agent_response)

    # 5️⃣ Envoi de la réponse à WhatsApp
    whatsapp_response = send_whatsapp_message(agent_response, data)

    try:
        whatsapp_response.raise_for_status()
        return {"message": "Message traité avec succès"}
    except requests.exceptions.RequestException as e:
        rich.print(f"Erreur lors de l'envoi du message WhatsApp : {e}")
        return {"message": "Erreur WhatsApp"}

async def handle_audio_message(data: Dict[str, str]) -> dict:
    """
    Gère le traitement des messages audio entrants.
    
    Args:
        data (dict): Dictionnaire contenant les informations du message utilisateur.

    Returns:
        dict: Message indiquant si le traitement a réussi ou non.
    """
    message_id = data["user_audio_id"]

    # 1️⃣ Vérifier si le message a déjà été traité
    if message_id in processed_messages:
        logger.info(f"Message déjà traité : {message_id}")
        return {"message": "Message déjà traité"}  # Ignore les doublons

    processed_messages.add(message_id)  # Marquer le message comme traité

    user_phone = data["user_phone_number"]
    user_name = data.get("user_profil_name", "Unknown")

    # 2️⃣ Vérifier/créer l'utilisateur
    user_id = get_or_create_user(phone=user_phone, name=user_name)

    # 3️⃣ Récupérer un ID de conversation valide
    conversation_id, new_conversation = get_or_create_conversation_id(user_id)
    
    # Si c'est une nouvelle conversation, envoyer un message d'introduction
    if new_conversation:
        send_whatsapp_message(message, data)

    # 4️⃣ Transcription de l'audio
    try:
        audio_text = process_audio_messages(audio_id=data["user_audio_id"])
    except requests.exceptions.ReadTimeout:
        save_message(conversation_id, "user", "Problème de connexion, veuillez réessayer svp")
        return f"Probleme de connexion"

    if not audio_text.strip():
        return {"message": "Audio non valide ou vide"}

    # Sauvegarder le message utilisateur
    save_message(conversation_id, "user", audio_text)

    # 5️⃣ Appel à l'agent pour générer une réponse
    agent_response = stream_chat(audio_text, conversation_id, data["user_profil_name"])

    # Sauvegarder la réponse de l'agent
    save_message(conversation_id, "bot", agent_response)

    # 6️⃣ Transformation en audio (TTS)
    try:
        audio_path = text_to_speech_with_gtts(text=agent_response, filename=f"./data/audio/{message_id}.mp3")
    except Exception as e:
        try:
            audio_path, _ = text_to_speech_with_openai(text=agent_response, filename=f"./data/audio/{message_id}.mp3")
        except Exception as e:
            logger.exception(f"Erreur TTS : {e}")
            return {"message": "Erreur de conversion texte en audio"}

    # 7️⃣ Upload du fichier audio
    audio_id = upload_audio_file(path=audio_path)

    # 8️⃣ Envoi de la réponse audio sur WhatsApp
    whatsapp_response = send_audio_message(to=data["user_phone_number"], audio_id=audio_id)

    try:
        whatsapp_response.raise_for_status()
        # logger.success(f"Réponse envoyée avec succès.")
        return {"message": "Message traité avec succès"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur lors de l'envoi du message WhatsApp : {e}")
        return {"message": "Erreur WhatsApp"}
    except Exception as e:
        logger.exception(f"Erreur inconnu -->> {e}")









   