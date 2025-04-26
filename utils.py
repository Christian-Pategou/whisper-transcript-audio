from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

_ = load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vector_store = Chroma(
    collection_name="imesy_chromadb_multi_allmini",
    embedding_function=embedding,
    persist_directory="./imesy_chromadb",
)

# vector_store = Chroma(
#     collection_name="josue_data",
#     embedding_function=embedding,
#     persist_directory="./../../Josue_chromadb",
# )

retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={ 'k':5, 'lambda_mult': 0.8, 'fetch_k':15},
)

model_ggl = ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash-exp", #"gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=3,
    temperature=0,
)

model_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=3
)  

pod_id = "emsuxxjl8xitpt"

model_ollama =  ChatOllama(
    model="gemma3:12b-it-q4_K_M",
    base_url=f"https://{pod_id}-11434.proxy.runpod.net",
    keep_alive=-1
)