from smolagents import CodeAgent, tool
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()


############################## agent ################################

@tool
def get_patient_id(name:str)-> str:
    """ ton rôle est de recuperer l'id d'un patient dans le fichier patients.csv à partir de son nom ou son prenom.
    Args:
        name: le nom ou le prénom du patient.
    return:
        id du patient
    
    """
    import pandas as pd
    patients = pd.read_csv('patients.csv')
    # consultations = pd.read_csv('consultations.csv')

    # Find the patient ID 
    patient_i1 = patients[patients['firstname'].str.lower()==name.lower()]['_id']
    patient_i2 = patients[patients['lastname'].str.lower()==name.lower()]['_id']
    if len(patient_i1) > 0:
        patient_id = patient_i1.values[0]
    else:
        try:
            patient_id=patient_i2.values[0]
        except IndexError as e:
            patient_id=f"Aucun patient avec le nom {name}"
    
    return patient_id

# @tool
# def get_patient_id_filter_by_doctor_id(name:str, id_doctor:str)-> str:
#     """ ton rôle est de recuperer l'id d'un patient dans le fichier patients.csv à partir de son nom ou son prenom.
#     Args:
#         name: le nom ou le prénom du patient.
#     return:
#         id du patient
    
#     """
    # import pandas as pd
    # patients = pd.read_csv('patients.csv')
    # # consultations = pd.read_csv('consultations.csv')

    # # Find the patient ID 
    # patient_i1 = patients[patients['firstname'].str.lower()==name.lower()]['_id']
    # patient_i2 = patients[patients['lastname'].str.lower()==name.lower()]['_id']
    # if len(patient_i1) > 0:
    #     patient_id = patient_i1.values[0]
    # else:
    #     try:
    #         patient_id=patient_i2.values[0]
    #     except IndexError as e:
    #         patient_id=f"Aucun patient avec le nom {name}"
    
    # return patient_id


def ana_agent(model):

    agent = CodeAgent(
        # system_prompt=prompt,
        tools=[get_patient_id],
        model=model,
        max_steps=5,
        additional_authorized_imports=["pandas", "os", "numpy", "datetime"],
        verbosity_level=2 ,
    )

    return agent 

############################## langchain response ################################


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
vector_store = Chroma(
    collection_name="imesy_chromadb",
    embedding_function=embedding,
    persist_directory="imest_chroma",  
)

retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={ 'k':15, 'lambda_mult': 0.3, 'fetch_k':30},
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":15}
)


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2
    
)
model_chain = ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash-exp", #"gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2
    
)

prompt_support = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es un assistant capable de repondre de facon claire et structurer a la question de l'utilisateur en te servant uniquement des informations mise à ta disposition.
            Tes reponses doivent etre bien formatées (au format mardown si possible).
            Voici le Context: \n\t{context}
            
                Pour les questions auxquelles tu n'as pas de réfenrence dans le document, reponds en disant qu'il ne t'ai possible de répondre à la question pour le moment.
            """,
        ),
        ("human", "{question}"),
    ]
)

retrieval_chain = (
    {"context": itemgetter("question") | retriever_mmr,
    "question": itemgetter("question")}
    | prompt_support
    | model_chain
    | StrOutputParser()
)
