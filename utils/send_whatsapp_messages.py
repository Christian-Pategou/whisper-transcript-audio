import requests
from dotenv import load_dotenv, find_dotenv
import os, rich
from logs import logger
from typing import Optional


# --------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------

load_dotenv(find_dotenv(".env"))


ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
RECIPIENT_WAID = os.getenv("RECIPIENT_WAID")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERSION = os.getenv("VERSION")

APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
API_URL = os.getenv("API_URL")

# --------------------------------------------------------------
# Send a template WhatsApp message
# --------------------------------------------------------------

def send_whatsapp_message() -> Optional[requests.Response]:
    """
    Envoie un message WhatsApp via l'API Cloud de Meta.

    Returns:
        Optional[requests.Response]: L'objet réponse de l'API en cas de succès, sinon `None`.

    Raises:
        requests.exceptions.RequestException: En cas d'échec de la requête HTTP.

    Exemple:
        >>> response = send_whatsapp_message()
        >>> print(response.status_code if response else "Échec de l'envoi")
    """

    url = f"{API_URL}/{VERSION}/{PHONE_NUMBER_ID}/messages"

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "messaging_product": "whatsapp",
        "to": RECIPIENT_WAID,
        "type": "template",
        "template": {
            "name": "Hello world!",
            "language": {"code": "fr_FR"}
        },
    }

    try:
        response = requests.post(url=url, json=data, headers=headers)
        response.raise_for_status()  # Vérifie si la requête a échoué

        logger.success(f"✅ Message WhatsApp envoyé avec succès ! Statut : {response.status_code}")
        return response

    except requests.exceptions.RequestException as e:
        logger.exception(f"❌ Erreur lors de l'envoi du message WhatsApp : {e}")
        return None


def format_data(to:str, message:str) -> dict:
    pass

if __name__ == "__main__":
    response = send_whatsapp_message()
    rich.print(response.json())
