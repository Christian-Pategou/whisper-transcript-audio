import sys
sys.path.append(".")
import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompt import (
    prompt_system_resume_fr, prompt_system_resume_en, prompt_system_diagnostic_fr, prompt_system_diagnostic_en,
    prompt_system_proposition, prompt_system_proposition_2_en, prompt_response_to_json, resume_consultation_final_prompt_fr, resume_consultation_final_prompt_en,
    prompt_system_proposition_2_fr, final_prompt_en, final_prompt_fr, prompt_consultation_resume_fr, prompt_consultation_resume_en,
    prompt_paraclinique_fr, prompt_paraclinique_en, prompt_clinique_fr, prompt_clinique_en, regflag_final_prompt_fr, regflag_final_prompt_en,
    prompt_format_paraclinique_en, prompt_format_paraclinique_fr, prompt_format_clinique_fr, prompt_format_clinique_en, prompt_format_prescription_fr, prompt_format_prescription_en
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
    temperature=0.2,
    max_retries=4,
    api_key=os.getenv("GROQ_API_KEY")
)

model_format = ChatGroq(
    model=os.getenv("GROQ_MODEL_NAME_2"),
    temperature=0,
    max_retries=4,
    api_key="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"     #os.getenv("GROQ_API_KEY")
)

def retrieval_resume_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_resume_fr
            | models
            | StrOutputParser()
        )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_resume_en
            | models
            | StrOutputParser()
        )

def retrieval_diagnostic_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_diagnostic_fr
            | models
            | StrOutputParser()
        )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_diagnostic_en
            | models
            | StrOutputParser()
        )

def retrieval_proposition_2_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_proposition_2_fr
            | models
            | JsonOutputParser()
        )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_system_proposition_2_en
            | models
            | JsonOutputParser()
        )

def chain_consultation_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return prompt_consultation_resume_fr | model_ggl | JsonOutputParser()
        case "en":
            return prompt_consultation_resume_en | model_ggl | JsonOutputParser()

def retrieval_format_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough(),
                "instruction" : RunnablePassthrough()}
                | final_prompt_fr
                | models 
                | StrOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough(),
            "instruction" : RunnablePassthrough()}
            | final_prompt_en
            | models 
            | StrOutputParser()
        )

def retrieval_regflag_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough(),
                "prescription_medecin" : RunnablePassthrough()}
                | regflag_final_prompt_fr
                | models
                | StrOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough(),
            "prescription_medecin" : RunnablePassthrough()}
            | regflag_final_prompt_en
            | models
            | StrOutputParser()
        )

def retrieval_resume_consultation_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough()}
                | resume_consultation_final_prompt_fr
                | models 
                | StrOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | resume_consultation_final_prompt_en
            | models
            | StrOutputParser()
        )

def retrieval_format_paraclinique_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough()}
                | prompt_format_paraclinique_fr
                | models 
                | JsonOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_format_paraclinique_en
            | models
            | JsonOutputParser()
        )

def retrieval_format_clinique_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough()}
                | prompt_format_clinique_fr
                | models 
                | JsonOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_format_clinique_en
            | models
            | JsonOutputParser()
        )

def retrieval_format_prescription_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
                {"input": RunnablePassthrough()}
                | prompt_format_prescription_fr
                | models 
                | JsonOutputParser()
            )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_format_prescription_en
            | models
            | JsonOutputParser()
        )


retrieval_proposition = (
    {"input": RunnablePassthrough()}
    | prompt_system_proposition
    | model_ggl
    | JsonOutputParser()
)

retrieval_proposition_groq = (
    {"input": RunnablePassthrough()}
    | prompt_system_proposition
    | model
    | JsonOutputParser()
)

retrieval_json = (
    {"input": RunnablePassthrough()}
    | prompt_response_to_json
    | model_ggl
    | JsonOutputParser()
)

retrieval_json_groq = (
    {"input": RunnablePassthrough()}
    | prompt_response_to_json
    | model
    | JsonOutputParser()
)

def retrieval_clinique_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
            {"input": RunnablePassthrough()}
            | prompt_clinique_fr
            | models
            | StrOutputParser()
        )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_clinique_en
            | models
            | StrOutputParser()
        )

def retrieval_paraclinique_lang(lang:str, models):
    print(f"langue detecté: {lang}")
    match lang:
        case "fr":
            return (
            {"input": RunnablePassthrough()}
            | prompt_paraclinique_fr
            | models
            | StrOutputParser()
        )
        case "en":
            return (
            {"input": RunnablePassthrough()}
            | prompt_paraclinique_en
            | models
            | StrOutputParser()
        )