from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from logger import logger
import os

load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGODB_URI") 
DATABASE_NAME = os.getenv("DATABASE_NAME") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME") 

# support_links_context = ""  # Variable globale (ou mieux, un cache partagé selon les besoins)

# --- Connexion et Récupération ---
client = None # Initialiser la variable client
support_links_context=""

def format_links_for_prompt(links):
    return "\n".join([f"- {entry['thematique']}: {entry['link']}" for entry in links])


def fetch_links_from_db():
    global support_links_context
    try:
        client = MongoClient(str(MONGO_URI))
        client.admin.command('ping')  # Forcer une tentative de connexion pour vérifier si elle réussit (optionnel mais recommandé)
        print("Connexion à MongoDB réussie !")

        links_db = client[str(DATABASE_NAME)]
        links_col = links_db[str(COLLECTION_NAME)]
        links = list(links_col.find({}, {"_id": 0}))

        support_links_context = format_links_for_prompt(links)
        logger.info(f"✅ Liens de formation rechargés ({len(links)} liens).")
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Liens de formation rechargés ({len(links)} liens).")
    except ConnectionFailure as e:
        logger.error(f"Erreur de connexion à MongoDB : {e}")
    except Exception as e:
        logger.exception(f"Une erreur inattendue est survenue : {e}")
    finally:
        # Toujours fermer la connexion quand vous avez terminé
        if client:
            client.close()
            # print("\nConnexion MongoDB fermée.")
            logger.success("Connexion MongoDB fermée.")