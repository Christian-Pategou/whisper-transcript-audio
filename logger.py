from loguru import logger
import os
from dotenv import load_dotenv
import yagmail


load_dotenv()
# --- Config email ---
EMAIL_SENDER = os.getenv("SENDER_EMAIL")
EMAIL_PASSWORD = os.getenv("SENDER_PASSWORD")  # pas le mot de passe principal !
EMAIL_RECEIVER = os.getenv("RECEIVER_EMAIL")
# Création dossier logs
os.makedirs("logs", exist_ok=True)

# Nettoyer les anciens handlers
logger.remove()

# Console
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | <cyan>{message}</cyan>"
)

# Fichier
logger.add(
    "logs/scheduler.log",
    rotation="1 week",
    retention="1 month",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Handler email (niveau ERROR+ seulement)
def send_email_log(record):
    try:
        s = record.split("|")
        # Extraire l'heure, le niveau et le message du log
        timestamp = s[0]
        level = s[1].strip() # Puisque c'est une erreur, on le fixe
        message = s[2].split("-")[-1].strip() # Suppression des espaces inutiles dans le message

        # Création du contenu de l'email
        content = f"""
Une erreur critique a été détectée :

Temps : {timestamp}
Niveau : {level}
Message : {message}
        """

        # Choisir le sujet de l'email en fonction du niveau de log
        if level == "ERROR":
            subject = f"[ALERTE IMESY] Erreur niveau {level}"
        elif level == "WARNING":
            subject = f"[AVERTISSEMENT IMESY] Avertissement niveau {level}"
        elif level == "INFO":
            subject = f"[INFO IMESY] Information niveau {level}"
        else:
            subject = f"[IMESY] Log niveau {level}"

        # Envoi de l'email
        yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
        # subject = f"[ALERTE IMESY] Erreur niveau {level}"
        yag.send(to=EMAIL_RECEIVER, subject=subject, contents=content)
        print(f"Email envoyé avec success pour le level {level}")

    except Exception as e:
        print(f"[WARNING] Impossible d'envoyer l'email : {e}")

logger.add(send_email_log, level="ERROR")

# # Handler pour WARNING
# logger.add(
#     send_email_log, 
#     level="WARNING",
#     filter=lambda record: record.split("|")[1].strip() == "WARNING",  # N'envoie un email que pour les warnings
#     format="{message}"
# )

# # Handler pour ERROR
# logger.add(
#     send_email_log, 
#     level="ERROR",
#     filter=lambda record: record.split("|")[1].strip() == "ERROR",  # N'envoie un email que pour les erreurs
#     format="{message}"
# )
# logger.info("Test d'envoi d'email d'erreur !")
if __name__ == "__main__":
    from db import support_links_context
    print(support_links_context)

