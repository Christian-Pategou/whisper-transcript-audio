from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Dict
import uvicorn
from dotenv import load_dotenv
from tools.funct import (   
    retrieval_resume, 
    retrieval_diagnostic,
    retrieval_paraclinique,
    retrieval_format,
    retrieval_clinique,
    retrieval_proposition_2,
    chain_consultation,
    retrieval_regflag,
    retrieval_resume_consultation,
    retrieval_format_clinique,
    retrieval_format_paraclinique,
    retrieval_format_prescription
)

class QuestionRequest(BaseModel):
    question: str
    model: str

class GenerateRequest(BaseModel):
    questions: list
    model: str

class EditTextRequest(BaseModel):
    input:str
    instruct:str
    model: str

class PrescriptionRequest(BaseModel):
    input:str
    prescription:str
    model: str

load_dotenv()

app = FastAPI()

# Ajoute le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet uniquement les origines spÃ©cifiÃ©es
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les mÃ©thodes HTTP (GET, POST, etc.)
    allow_headers=["*"],   # Permet tous les en-tÃªtes
)

@app.post("/get_resume/")
async def get_response(request: QuestionRequest):
    try:
        # Appel du modèle pour obtenir la réponse
        response = retrieval_resume.invoke(request.question)
        
        return {"response": response}
    except Exception as e:
        print(request.question)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clinique/")
async def get_clinique(request: QuestionRequest):
    try:
        response = retrieval_clinique.invoke(request.question)
        return response
        # res_json = retrieval_json.invoke(response)
        # return dict({
        #     "respponse": response,
        #     "response_json" : res_json
        # })
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/paraclinique/")
async def get_paraclinique(request: QuestionRequest):
    try:
        response = retrieval_paraclinique.invoke(request.question)
        return response
        # res_json = retrieval_json.invoke(response)
        # return dict({
        #     "respponse": response,
        #     "response_json" : res_json
        # })
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_diagnostic/")
async def get_diagnostic(request: QuestionRequest):
    try:
        response = retrieval_diagnostic.invoke(request.question)
        return response
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_proposition/")
async def get_proposition(request: QuestionRequest):
    try:
        response = retrieval_proposition_2.invoke(request.question)
        return response
        # res_json = retrieval_json.invoke(response)
        # return dict({
        #     "respponse": response,
        #     "response_json" : res_json
        # })
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_consultation/")
async def get_consultation(request: QuestionRequest):
    try:
        response = chain_consultation.invoke(request.question) 
        return response
        # res_json = retrieval_json.invoke(response)
        # return dict({
        #     "respponse": response,
        #     "response_json" : res_json
        # })
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"
@app.post("/get_transcirpt/")
async def get_transcript(resquest:QuestionRequest):
    try:
        model = whisper.load_model("small")
        with open("/audio1.wav", "wb") as file:
            file.write(resquest.question)
        transcrib = model.transcribe(audio="./audio1.wav")
        text = transcrib["text"]
        return text
    except Exception as e:
        print(f"\n\nerror occured \t\t{e}")


@app.post("/format-text/")
async def format_text(request: EditTextRequest):
    try:
        return retrieval_format.invoke([request.input, request.instruct]).replace("\\n", "\n")
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"

@app.post("/reg_flag/")
def reg_flag(request: PrescriptionRequest) -> str:
    try:
        return retrieval_regflag.invoke([request.input, request.prescription]).replace("\n", "")
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"

@app.post("/format_prescription/")
def format_prescription(request: QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_prescription.invoke(request.question))
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"


@app.post("/format_paraclinique/")
def format_paraclinique(request: QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_paraclinique.invoke(request.question))
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"


@app.post("/format_clinique/")
def format_clinique(request: QuestionRequest) -> dict:
    try:
        resp = retrieval_format_clinique.invoke(request.question)
        return dict(resp)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"

@app.post("/summarize_consultation/")
def summarize_consultation(request: QuestionRequest) -> str:
    try:
        return retrieval_resume_consultation.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return "Rate Limit Exceeted"
            case _ :
                return f"{e}"
