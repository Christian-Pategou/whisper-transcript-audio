from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
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