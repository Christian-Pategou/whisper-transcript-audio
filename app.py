from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import whisper, os
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tools.funct import (
    retrieval_resume, retrieval_resume_groq, 
    retrieval_diagnostic, retrieval_diagnostic_groq,
    retrieval_paraclinique, retrieval_paraclinique_groq,
    retrieval_format, retrieval_format_groq,
    retrieval_clinique, retrieval_clinique_groq,
    retrieval_proposition_2, retrieval_proposition_2_groq,
    chain_consultation, chain_consultation_groq,
    retrieval_regflag, retrieval_regflag_groq,
    retrieval_resume_consultation, retrieval_resume_consultation_groq,
    retrieval_format_clinique, retrieval_format_clinique_groq,
    retrieval_format_paraclinique,retrieval_format_paraclinique_groq,
    retrieval_format_prescription, retrieval_format_prescription_groq,
)

class QuestionRequest(BaseModel):
    question: str

class PrescriptionRequest(BaseModel):
    input:str
    prescription:str
    model: str

class EditTextRequest(BaseModel):
    input:str
    instruct:str
    model: str


# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

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
        response = retrieval_resume_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_resume_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_resume.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_resume.invoke(request.question)
                case _ :
                    return retrieval_resume_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_resume.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_resume_groq.invoke(request.question)
            case _ :
                return retrieval_resume.invoke(request.question)


@app.post("/clinique/")
async def get_clinique(request: QuestionRequest):
    try:
        response = retrieval_clinique_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_clinique_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_clinique.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_clinique.invoke(request.question)
                case _ :
                    return retrieval_clinique_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_clinique.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_clinique_groq.invoke(request.question)
            case _ :
                return retrieval_clinique.invoke(request.question)
    

@app.post("/paraclinique/")
async def get_paraclinique(request: QuestionRequest):
    try:
        response = retrieval_paraclinique_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_paraclinique_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_paraclinique.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_paraclinique.invoke(request.question)
                case _ :
                    return retrieval_paraclinique_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_paraclinique.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_paraclinique_groq.invoke(request.question)
            case _ :
                return retrieval_paraclinique.invoke(request.question)

@app.post("/get_diagnostic/")
async def get_diagnostic(request: QuestionRequest):
    try:
        response = retrieval_diagnostic_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_diagnostic_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_diagnostic.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_diagnostic.invoke(request.question)
                case _ :
                    return retrieval_diagnostic_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_diagnostic.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_diagnostic_groq.invoke(request.question)
            case _ :
                return retrieval_diagnostic.invoke(request.question)
    

@app.post("/get_proposition/")
async def get_proposition(request: QuestionRequest):
    try:
        response = retrieval_proposition_2_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_proposition_2_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_proposition_2.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_proposition_2.invoke(request.question)
                case _ :
                    return retrieval_proposition_2_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_proposition_2.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_proposition_2_groq.invoke(request.question)
            case _ :
                return retrieval_proposition_2.invoke(request.question)


@app.post("/get_consultation/")
async def get_consultation(request: QuestionRequest):
    try:
        response = chain_consultation_groq.invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return chain_consultation_groq.invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return chain_consultation.invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return chain_consultation.invoke(request.question)
                case _ :
                    return chain_consultation_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return chain_consultation.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return chain_consultation_groq.invoke(request.question)
            case _ :
                return chain_consultation.invoke(request.question)


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


@app.post("/format_text/")
async def format_text(request: EditTextRequest):
    try:
        return retrieval_format_groq.invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return retrieval_format.invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
            case _ :
                try:
                    return retrieval_format_groq.invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return retrieval_format.invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
                        case _ :
                            return retrieval_format_groq.invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")

@app.post("/reg_flag/")
def reg_flag(request: PrescriptionRequest) -> str:
    try:
        return retrieval_regflag_groq.invoke([request.input, request.prescription_medecin]).replace("\n", "").replace("```", "")
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return retrieval_regflag.invoke([request.input, request.prescription_medecin]).replace("\n", "").replace("```", "")
            case _ :
                try:
                    return retrieval_regflag_groq.invoke([request.input, request.prescription_medecin]).replace("\n", "")
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return retrieval_regflag.invoke([request.input, request.prescription_medecin]).replace("\n", "").replace("```", "")
                        case _ :
                            return retrieval_regflag_groq.invoke([request.input, request.prescription_medecin]).replace("\n", "").replace("```", "")

@app.post("/format_prescription/")
def format_prescription(request:QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_prescription_groq.invoke(request.question))
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return dict(retrieval_format_prescription.invoke(request.question))
            case _ :
                try:
                    return dict(retrieval_format_prescription_groq.invoke(request.question))
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return dict(retrieval_format_prescription.invoke(request.question))
                        case _ :
                            return dict(retrieval_format_prescription_groq.invoke(request.question))


@app.post("/format_paraclinique/")
def format_paraclinique(request:QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_paraclinique_groq.invoke(request.question))
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                dict(retrieval_format_paraclinique.invoke(request.question))
            case _ :
                try:
                    return dict(retrieval_format_paraclinique.invoke(request.question))
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            dict(retrieval_format_paraclinique.invoke(request.question))
                        case _ :
                            dict(retrieval_format_paraclinique_groq.invoke(request.question))


@app.post("/format_clinique/")
def format_clinique(request:QuestionRequest) -> dict:
    try:
        resp = retrieval_format_clinique_groq.invoke(request.question)
        return dict(resp)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                resp = retrieval_format_clinique.invoke(request.question)
                return dict(resp)
            case _ :
                try:
                    resp = retrieval_format_clinique.invoke(request.question)
                    return dict(resp)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            resp = retrieval_format_clinique.invoke(request.question)
                            return dict(resp)
                        case _ :
                            resp = retrieval_format_clinique_groq.invoke(request.question)
                            return dict(resp)

@app.post("/summarize_consultation/")
def summarize_consultation(request:QuestionRequest) -> str:
    try:
        return retrieval_resume_consultation_groq.invoke(request.question)
    except Exception as e:
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                 return retrieval_resume_consultation.invoke(request.question)
            case _ :
                try:
                    return retrieval_resume_consultation.invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                             return retrieval_resume_consultation.invoke(request.question)
                        case _ :
                             return retrieval_resume_consultation_groq.invoke(request.question)

