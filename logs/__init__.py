import os
import sys
from loguru import logger
from dotenv import load_dotenv
import rich

load_dotenv()

# Définir votre format ici
FORMAT = os.getenv("FORMAT")

logger.remove()  # Supprime les handlers par défaut

try:
    logger.configure(
        handlers=[
            {
                "sink": r"../data/log/app.log",
                "format": FORMAT,
                "level": "WARNING",
                "rotation": "500 MB",
                "retention": "100 days",
            },
            {
                "sink": sys.stderr,
                "format": FORMAT,
                "level": "DEBUG",
            }
        ]
    )
except Exception as e:
    rich.print(f"Erreur lors de la configuration des logs : {e}")
    # Gérer l'erreur, par exemple, en utilisant un logging de base
    import logging
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Erreur de configuration de Loguru : {e}")
    # On peut aussi sortir du programme proprement si la journalisation est critique
    sys.exit(1)

def traiter_requete(utilisateur_nom, utilisateur_id, requete):
    logger.info(f"[{utilisateur_nom} - {utilisateur_id}] Début du traitement de la requête : {requete}")
    # ... Votre code de traitement de requête ...
    logger.debug(f"[{utilisateur_nom} - {utilisateur_id}] Informations de débogage spécifiques à la requête.")
    logger.warning(f"[{utilisateur_nom} - {utilisateur_id}] Un avertissement s'est produit lors du traitement.")
    logger.info(f"[{utilisateur_nom} - {utilisateur_id}] Fin du traitement de la requête.")

