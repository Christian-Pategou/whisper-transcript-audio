from utils import vector_store
from typing_extensions import TypedDict, List, Dict, Any
from langchain_core.documents import Document
from chains import (
    chain_retrieval_grader_document,
    chain_asnwer_question,
    chain_grader_question_answer,
    chain_rewriter_question,
)
from utils import model_ggl, model_groq

from typing import Literal
from chains import GradeAnswer

import smtplib
from email.message import EmailMessage 

from dotenv import load_dotenv
import os

load_dotenv("./../.env")

SMTP_SERVER = os.getenv("SMTP_SERVER") 
SMTP_PORT = os.getenv("SMTP_PORT") 
SENDER_EMAIL = os.getenv("SENDER_EMAIL") 
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") 
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL") 


# definir l'état de l'object
class GraphState(TypedDict):
    """
    Représente l'état de notre graphe.

    Args :
        question : question
        generation : Réponse du LLM
        documents : contexte lié à la question
        max_iter : nombre max d'itération
    """
    max_iter: int
    question: str
    documents: List[Document]
    answer: str


# fonction poour recuperer les documents pertinents
def retrieve_node(state:GraphState) -> GraphState:
    """
    Retrieve documents

    Args:
        state (dict): The current graph GraphState

    Returns:
        state (dict): Nouvelle clé ajoutée à l'état, documents, qui contient les documents récupérés
    """
    print("---RETRIEVE---")
    question = state["question"]
    

    # Retrieval
    documents = vector_store.similarity_search(query=question, k=5)
    print(f"document tourve: {documents}")
    return {"documents": documents, "question": question, "max_iter": state["max_iter"]}



def grade_documents_node(state:GraphState) -> GraphState:
    """
    Détermine si les documents extraits sont pertinents par rapport à la question.

    Args :
        state (dict) : L'état actuel du graphe

    Returns :
        state (dict) : Met à jour la clé des documents avec uniquement les documents pertinents filtrés.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    def essaie(model, d, f):
            score = chain_retrieval_grader_document(model=model).invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "oui":
                print("---GRADE: DOCUMENT RELEVANT---")
                f.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
    # Score each doc
    filtered_docs = []
    for d in documents:
        try:
            print("model groq")
            essaie(model_groq, d, filtered_docs)
        except Exception as e:
            print(f"\nUne erreur est survenue: GROQ --->>>> {e}\n")
            try:
                print("model google")
                essaie(model_ggl, d, filtered_docs)
            except Exception as e:
                print(f"\nUne erreur est survenue: GOOGLE --->>>> {e}\n")
    print(f"doc filtrer {filtered_docs}")
    return {"documents": filtered_docs, "question": question}



def generate_node(state:GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (dict): The current graph GraphState

    Returns:
        state (dict): New key added to GraphState, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    max_iter = state["max_iter"]

    # RAG generation
    def answers(model):
        return chain_asnwer_question(model).invoke({"context": documents, "question": question})
    try:
        print("model google")
        answer = answers(model_ggl)
    except Exception as e:
        print(f"\nUne erreur est survenue: GOOGLE --->>>> {e}\n")
        try:
            print("model groq")
            answer = answers(model_groq)
        except Exception as e:
            print(f"\nUne erreur est survenue: GROQ --->>>> {e}\n")
    print(f"llm response {answer}")
    return {"question": question, "answer": answer, "max_iter": max_iter}


def router_answer_cond(state:GraphState) -> Literal["bon", "mauvais", "humain"]:
    print("---- EVALUATION DE LA REPONSE GENERE ----")
    
    def check_answer(model) -> GradeAnswer:
        answer_grader = chain_grader_question_answer(model)
        return answer_grader.invoke({"question": state["question"], "answer": state["answer"]})
    
    try:
        print("model google")
        grader = check_answer(model_ggl)
    except Exception as e:
        print(f"\nUne erreur est survenue: GOOGLE --->>>> {e}\n")
        try:
            print("model groq")
            grader = check_answer(model_groq)
        except Exception as e:
            print(f"\nUne erreur est survenue: GROQ --->>>> {e}\n")

    match grader.answer_score:
        case "bon":
            print("good")
            return "good"
        case "mauvais":
            print("bad")
            return "bad"
        case _:
            print("humain")
            return "humain"
        

def transform_query_node(state:GraphState) -> GraphState:
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph GraphState

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    max_iter = state["max_iter"]

    print(f" Voici la reponse proposée: {state['answer']}")

    # Re-write question
    try:
        print("model google")
        better_question = chain_rewriter_question(model_ggl).invoke({"question": question})
    except Exception as e:
        print(f"\nUne erreur est survenue: GOOGLE --->>>> {e}\n")
        try:
            print("model groq")
            better_question = chain_rewriter_question(model_groq).invoke({"question": question})
        except Exception as e:
            print(f"\nUne erreur est survenue: GROQ --->>>> {e}\n")
    print(f"\n Voici la question reformulée: {better_question}\n")
    return {"question": better_question, "max_iter": max_iter + 1}


def send_email_or_retry_cond(state:GraphState) -> Literal["not_email", "email"]:
    max_iter = state["max_iter"]

    if max_iter < 2:
        print("---DECISION: RETRY RETRIEVER AND GENERATION---")
        return "not_email"
    else:
        print("---SEND EMAIL---")
        return "email"
    

def send_email_to_support_node(state: GraphState) -> GraphState: 
    """
    Envoie un email au support humain avec la question de l'utilisateur.
    Retourne un dictionnaire indiquant le succès ou l'échec et un message pour l'utilisateur.
    Args:
        state (dict): The current graph GraphState

    Returns:
        state (dict): Updates question key with a re-phrased answer
    """
    print("--- ENTER TO SEND EMAIL FUNCTION ---")
    question = state["question"] 

    # Créer l'objet EmailMessage
    message = EmailMessage()
    message['Subject'] = f"Question non répondue nécessitant intervention : {question[:50]}..."
    message['From'] = SENDER_EMAIL
    message['To'] = RECEIVER_EMAIL

    # Corps de l'email
    body = f"""Bonjour,

    Ceci est un message automatique du système de support IMESY.

    Impossible de répondre automatiquement à la question suivante posée par un utilisateur :

    "{question}"

    Merci de prendre le relais.

    Cordialement,
    Système de Support Automatisé IMESY
"""
    # Définir le contenu du message (texte simple)
    message.set_content(body, subtype='plain', charset='utf-8') # Assure l'encodage correct

    print(f"Envoi de l'email à {RECEIVER_EMAIL}...")
    print(f"SMTP_SERVER: {SMTP_SERVER}")
    print(f"TYPE_SMTP_PORT: {type(SMTP_PORT)}")
    print(f"SMTP_PORT: {SMTP_PORT}")
    print(f"SENDER_EMAIL: {SENDER_EMAIL}")

    try:
        SMTP_PORT = int(SMTP_PORT)
    except (ValueError, TypeError):
        print(f"ERREUR: Le port SMTP '{SMTP_PORT}' n'est pas un nombre valide.")
 

    try:
        # Utiliser un contexte `with` pour assurer la fermeture de la connexion
        with smtplib.SMTP(host=SMTP_SERVER, port=SMTP_PORT) as server:
            server.ehlo()  # Saluer le serveur
            server.starttls()  # Activer le chiffrement TLS
            server.ehlo()  # Re-saluer après TLS
            print(f"Authentification avec {SENDER_EMAIL}...")
            # Utiliser la variable SENDER_EMAIL (string) pour l'authentification
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            print(f"Envoi de l'email à {RECEIVER_EMAIL}...")
            # Utiliser send_message pour les objets EmailMessage
            server.send_message(message)
            # Alternative si vous utilisiez MIMEMultipart:
            # server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())
            print(f"---- EMAIL SENT SUCCESSFULLY TO {RECEIVER_EMAIL} ----")
            # Retourner un message pour l'utilisateur final et potentiellement un statut
        return {"answer": "Votre question a bien été transmise à notre équipe de support. Nous vous reviendrons dans les plus brefs délais."}

    except smtplib.SMTPAuthenticationError:
        print("ERROR: Échec de l'authentification SMTP.")
        print("Vérifiez l'e-mail/mot de passe.")
        print("Si vous utilisez Gmail avec 2FA, assurez-vous d'utiliser un 'Mot de passe d'application'.")
        print("Vérifiez également si l'accès aux applications moins sécurisées est nécessaire (NON RECOMMANDÉ).")
        return {"answer": "Désolé, une erreur technique (authentification) a empêché la transmission de votre question. Veuillez réessayer plus tard ou contacter le support directement."}
    except smtplib.SMTPConnectError:
        print(f"ERROR: Échec de la connexion au serveur SMTP : {SMTP_SERVER}:{SMTP_PORT}")
        return {"answer": "Désolé, une erreur technique (connexion serveur) a empêché la transmission de votre question. Veuillez réessayer plus tard ou contacter le support directement."}
    except smtplib.SMTPServerDisconnected:
         print(f"ERROR: Déconnexion inattendue du serveur SMTP {SMTP_SERVER}")
         return {"answer": "Désolé, une erreur technique (déconnexion serveur) a empêché la transmission de votre question. Veuillez réessayer plus tard ou contacter le support directement."}
    except Exception as e:
        # Capturer toute autre exception (ex: problème réseau, erreur inattendue)
        print(f"ERROR: Une erreur inattendue est survenue lors de l'envoi de l'e-mail : {e}")
        import traceback
        traceback.print_exc() # Affiche la trace complète pour le débogage
        return {"answer": f"Désolé, une erreur technique ({type(e).__name__}) a empêché la transmission de votre question. Veuillez réessayer plus tard ou contacter le support directement."}