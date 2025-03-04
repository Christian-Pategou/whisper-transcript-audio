import sys
sys.path.append(".")
import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompt import (
    prompt_system_resume, prompt_system_diagnostic, 
    prompt_system_proposition, prompt_format, prompt_response_to_json, 
    prompt_system_proposition_2, final_prompt, prompt_consultation_resume,
    prompt_paraclinique, prompt_clinique, regflag_final_prompt, resume_consultation_final_prompt,
    prompt_format_paraclinique, prompt_format_clinique, prompt_format_prescription
)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv()


chat_model = ChatOllama(
    model = "llama3.2:3b",
    base_url= os.getenv("OLLAMA_BASE_URL", "https://recette-apps.pategou.com/ollama/") #"http://localhost:11434/" 
)

model_ggl = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  #"gemini-1.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=4,
    # other params...
)
model = ChatGroq(
    model=os.getenv("GROQ_MODEL_NAME_2"),
    temperature=0.3,
    max_retries=4,
    api_key=os.getenv("GROQ_API_KEY")
)

model_format = ChatGroq(
    model=os.getenv("GROQ_MODEL_NAME_2"),
    temperature=0,
    max_retries=4,
    api_key="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"     #os.getenv("GROQ_API_KEY")
)

retrieval_resume = (
    {"input": RunnablePassthrough()}
    | prompt_system_resume
    | model
    | StrOutputParser()
)

retrieval_diagnostic = (
    {"input": RunnablePassthrough()}
    | prompt_system_diagnostic
    | model
    | StrOutputParser()
)

retrieval_proposition = (
    {"input": RunnablePassthrough()}
    | prompt_system_proposition
    | model
    | JsonOutputParser()
)

retrieval_proposition_2 = (
    {"input": RunnablePassthrough()}
    | prompt_system_proposition_2
    | model_ggl # model # 
    | JsonOutputParser()
)

chain_consultation = prompt_consultation_resume | model | JsonOutputParser()
# chain_consultation = prompt_consultation_resume | model_ggl | JsonOutputParser()

retrieval_format = (
    {"input": RunnablePassthrough(),
     "instruction" : RunnablePassthrough()}
    | final_prompt
    | model # model_ggl
    | StrOutputParser()
)

retrieval_regflag = (
    {"input": RunnablePassthrough(),
     "prescription_medecin" : RunnablePassthrough()}
    | regflag_final_prompt
    | model # model_ggl
    | StrOutputParser()
)

retrieval_resume_consultation = (
    {"input": RunnablePassthrough()}
    | resume_consultation_final_prompt
    | model_format
    | StrOutputParser()
)
retrieval_format_clinique = (
    {"input": RunnablePassthrough()}
    | prompt_format_clinique
    | model_format
    | JsonOutputParser()
)

retrieval_format_paraclinique = (
    {"input": RunnablePassthrough()}
    | prompt_format_paraclinique
    | model_format
    | JsonOutputParser()
)


retrieval_format_prescription = (
    {"input": RunnablePassthrough()}
    | prompt_format_prescription
    | model_format
    | JsonOutputParser()
)


retrieval_json = (
    {"input": RunnablePassthrough()}
    | prompt_response_to_json
    | model
    | JsonOutputParser()
)

retrieval_clinique = (
    {"input": RunnablePassthrough()}
    | prompt_clinique
    | model_format
    | JsonOutputParser()
)

retrieval_paraclinique = (
    {"input": RunnablePassthrough()}
    | prompt_paraclinique
    | model_format
    | JsonOutputParser()
)