from fastapi import FastAPI
from smolagents import HfApiModel, LiteLLMModel
from agent import ana_agent as imesy_agent, retrieval_chain
from prompts import prompt

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

model_hf = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.getenv('HF_TOKEN')
)

model_gemini = LiteLLMModel(
    model_id="gemini/gemini-2.0-flash-lite-preview-02-05",   #"gemini/gemini-2.0-flash-exp", #"openrouter/google/gemini-2.0-pro-exp-02-05:free"
    api_key = os.getenv('GOOGLE_API_KEY')
)

model_chain = ChatGoogleGenerativeAI(
    model= "gemini-2.0-flash-exp", #"gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=2
    
)
app = FastAPI()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
Tu es un assistant intelligent qui doit router une question à l'un des deux services :
- 'support' pour les questions sur l'utilisation de l'application. L'application s'appelle IMESY. c'est une aapp web permettant aux medecins de simplifier le processus de consultation.
- 'agent_medical' pour les questions concernant dont la reponse necessite de faire une recherche dans la base de données.

Question: {question}
Catégorie (support ou agent_medical) ?
Réponds uniquement par "support" ou "agent_medical".
"""),
("human", "{question}")
    ]
)


router = prompt_template | model_chain

@app.post("/imesy-router/")
async def imesy_router(question:str):
    res = await router.ainvoke(question)
    if res.content == "agent_medical":
        print("enter to agent bot")
        agent = imesy_agent(model_gemini)
        try:
            response = agent.run(
                prompt + "\nq" + question,
                additional_args=dict(source_file=["patients.csv", "consultations.csv"])
            )
            
        except Exception as e:
            print(f"une erreur  s'est produite:\n\n {e}")

    elif res.content == "support":
        print("enter to support bot")
        response = await retrieval_chain.ainvoke({"question":question})
        
    else:
        return f"une erreur s'est produite \n {res}"
    return response



