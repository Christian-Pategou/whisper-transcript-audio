from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.exceptions import OutputParserException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import rich, re, json
from langdetect import detect
from tools.funct import (
    retrieval_resume_lang, retrieval_clinique_lang, retrieval_paraclinique_lang, retrieval_resume_consultation_lang,
retrieval_diagnostic_lang, retrieval_proposition_2_lang, chain_consultation_lang, retrieval_format_lang,
retrieval_regflag_lang, retrieval_format_prescription_lang, retrieval_format_paraclinique_lang, retrieval_format_clinique_lang
)
from tools.funct import model_ggl, model


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



def extract_json_from_text(text):
    """
    Extrait le contenu JSON situé entre des balises de code Markdown ``` dans le texte.
    
    Args:
        text (str): Le texte contenant le JSON.
    
    Returns:
        dict | list: Le contenu JSON extrait sous forme de dictionnaire ou liste Python.
    """
    try:
        # Cherche le bloc entre les balises ``` (optionnellement avec json précisé)
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if match:
            json_text = match.group(1)
            return json.loads(json_text)
        else:
            raise ValueError("Aucun bloc JSON trouvé dans le texte.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur lors du parsing JSON : {e}")


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

output_parser_exception = {
    "anamnèse": "",
    "examen": {
        "paraclinique": [],
        "clinique": []
    },
    "diagnostic": "",
    "traitement": {
        "laboratoire": [],
        "imagerie": [],
        "ophtalmologie": [],
        "medicaments": [],
        "recommendation": "",
        "autres": []
    }
}

@app.post("/get_resume/")
async def get_response(request: QuestionRequest)-> str:
    try:
        return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
        
    except HTTPException as e:
        try:
            return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except OutputParserException as e:
            try:
                return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
            except OutputParserException:
                return output_parser_exception
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_resume_lang(detect(request.question), model_ggl).invoke(request.question)
                    except OutputParserException:
                        try:
                            return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
                        except Exception as e:
                            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                            return output_parser_exception
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_resume_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_resume_lang(detect(request.question), model_ggl).invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_resume_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return retrieval_resume_lang(detect(request.question), model_ggl).invoke(request.question)

@app.post("/clinique/")
async def get_clinique(request: QuestionRequest)-> dict:
    try:
        return retrieval_clinique_lang(detect(request.question), model).invoke(request.question)
    except HTTPException as e:
        try:
            return retrieval_clinique_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"\n\nProbleme de connexion: {e}\n\n"
            # raise HTTPException(status_code=500, detail=str(e))
        except OutputParserException:
            try:
                return retrieval_clinique_lang(detect(request.question), model).invoke(request.question)
            except OutputParserException:
                return retrieval_clinique_lang(detect(request.question), model_ggl).invoke(request.question)
            except Exception as e:
                rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                return {"clinique"}
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_clinique_lang(detect(request.question), model_ggl).invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_clinique_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return retrieval_clinique_lang(detect(request.question), model).invoke(request.question)
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_clinique_lang(detect(request.question), model_ggl).invoke(request.question)
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_clinique_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return retrieval_clinique_lang(detect(request.question), model_ggl).invoke(request.question)
    
@app.post("/paraclinique/")
async def get_paraclinique(request: QuestionRequest)-> dict:
    try:
        return retrieval_paraclinique_lang(detect(request.question), model).invoke(request.question)
    except HTTPException as e:
        try:
            return retrieval_paraclinique_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question)
                    except Exception as e:
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return retrieval_paraclinique_lang(detect(request.question), model).invoke(request.question)
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question)
                except Exception as e:
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_paraclinique_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return retrieval_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question)

@app.post("/get_diagnostic/")
async def get_diagnostic(request: QuestionRequest)-> str:
    try:
        response = retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except OutputParserException:
            try:
                return retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
            except Exception as e:
                rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                return "please try again" 
        except Exception as e:
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
                    except OutputParserException:
                        try:
                            return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
                        except Exception as e:
                            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                            return "please try again"
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
    except OutputParserException:
        try:
            return retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
            return 'please try again'
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
                except OutputParserException:
                    try:
                        return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                        return "please try again"
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}")
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_diagnostic_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return retrieval_diagnostic_lang(detect(request.question), model_ggl).invoke(request.question)
    

@app.post("/get_proposition/")
async def get_proposition(request: QuestionRequest)-> dict:
    try:
        response = retrieval_proposition_2_lang(detect(request.question), model).invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return retrieval_proposition_2_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except OutputParserException:
            try:
                return retrieval_proposition_2_lang(detect(request.question), model).invoke(request.question)
            except Exception as e:
                rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                return output_parser_exception
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)
                    except OutputParserException:
                        try:
                            return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)
                        except Exception as e:
                            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                            return output_parser_exception
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return retrieval_proposition_2_lang(detect(request.question), model).invoke(request.question)
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)
                except OutputParserException:
                    try:
                        return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        return output_parser_exception
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return retrieval_proposition_2_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return retrieval_proposition_2_lang(detect(request.question), model_ggl).invoke(request.question)


@app.post("/get_consultation/")
async def get_consultation(request: QuestionRequest)-> dict:
    try:
        response = chain_consultation_lang(detect(request.question), model).invoke(request.question)
        return response
    except HTTPException as e:
        try:
            return chain_consultation_lang(detect(request.question), model).invoke(request.question)
        except HTTPException as e:
            return f"Probleme de connexion: {e}"
            # raise HTTPException(status_code=500, detail=str(e))
        except  OutputParserException as e:
            try:
                return chain_consultation_lang(detect(request.question), model).invoke(request.question)
            except:
                rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                return output_parser_exception
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            match e.status_code:
                case 400:
                    return "Organization restricted (Groq)"
                case 429:
                    try:
                        return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                    except  OutputParserException as e:
                        try:
                            return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                        except:
                            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                            return output_parser_exception
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        match e.status_code:
                            case 400:
                                return "Organization restricted (Google)"
                            case 429:
                                return "Rate Limit Exceeted (Google)"
                            case _ :
                                return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                case _ :
                    return chain_consultation_lang(detect(request.question), model).invoke(request.question)
    except  OutputParserException as e:
        try:
            return chain_consultation_lang(detect(request.question), model).invoke(request.question)
        except:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            return output_parser_exception
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted (Groq)"
            case 429:
                try:
                    return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                except  OutputParserException as e:
                    try:
                        return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                    except:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        return output_parser_exception
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted (Google)"
                        case 429:
                            return "Rate Limit Exceeted (Google)"
                        case _ :
                            return chain_consultation_lang(detect(request.question), model).invoke(request.question)
            case _ :
                return chain_consultation_lang(detect(request.question), model_ggl).invoke(request.question)


@app.post("/format_text/")
async def format_text(request: EditTextRequest)-> str:
    try:
        return retrieval_format_lang(detect(request.question), model).invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return retrieval_format_lang(detect(request.question), model_ggl).invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
            case _ :
                try:
                    return retrieval_format_lang(detect(request.question), model).invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return retrieval_format_lang(detect(request.question), model_ggl).invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")
                        case _ :
                            return retrieval_format_lang(detect(request.question), model).invoke([request.input, request.instruct]).replace("\\n", "\n").replace("```", "")

@app.post("/reg_flag/")
def reg_flag(request: PrescriptionRequest) -> str:
    try:
        return retrieval_regflag_lang(detect(request.question), model).invoke([request.input, request.prescription]).replace("\n", "").replace("```", "")
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                return retrieval_regflag_lang(detect(request.question), model_ggl).invoke([request.input, request.prescription]).replace("\n", "").replace("```", "")
            case _ :
                try:
                    return retrieval_regflag_lang(detect(request.question), model).invoke([request.input, request.prescription]).replace("\n", "").replace("```", "")
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return retrieval_regflag_lang(detect(request.question), model_ggl).invoke([request.input, request.prescription]).replace("\n", "").replace("```", "")
                        case _ :
                            return retrieval_regflag_lang(detect(request.question), model).invoke([request.input, request.prescription]).replace("\n", "").replace("```", "").replace("```", "")

@app.post("/format_prescription/")
def format_prescription(request:QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_prescription_lang(detect(request.question), model).invoke(request.question))
    except OutputParserException:
        try:
            return dict(retrieval_format_prescription_lang(detect(request.question), model).invoke(request.question))
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            return output_parser_exception
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                try:
                    return dict(retrieval_format_prescription_lang(detect(request.question), model_ggl).invoke(request.question))
                except OutputParserException:
                    try:
                        return dict(retrieval_format_prescription_lang(detect(request.question), model_ggl).invoke(request.question))
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        return output_parser_exception
            case _ :
                try:
                    return dict(retrieval_format_prescription_lang(detect(request.question), model).invoke(request.question))
                except OutputParserException:
                    try:
                        return dict(retrieval_format_prescription_lang(detect(request.question), model).invoke(request.question))
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        return output_parser_exception
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return dict(retrieval_format_prescription_lang(detect(request.question), model_ggl).invoke(request.question))
                        case _ :
                            return dict(retrieval_format_prescription_lang(detect(request.question), model).invoke(request.question))


@app.post("/format_paraclinique/")
def format_paraclinique(request:QuestionRequest) -> dict:
    try:
        res = retrieval_format_paraclinique_lang(detect(request.question), model).invoke(request.question)
        return dict(res)
        # return dict(extract_json_from_text(res))
    except OutputParserException:
        try:
            return dict(retrieval_format_paraclinique_lang(detect(request.question), model).invoke(request.question))
        except OutputParserException:
            return  dict(retrieval_format_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question))
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            return output_parser_exception
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                try:
                    return dict(retrieval_format_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question))
                except OutputParserException:
                    return dict(retrieval_format_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question))
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    return output_parser_exception
            case _ :
                try:
                    return dict(retrieval_format_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question))
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            return dict(retrieval_format_paraclinique_lang(detect(request.question), model_ggl).invoke(request.question))
                        case _ :
                            return dict(retrieval_format_paraclinique_lang(detect(request.question), model).invoke(request.question))


@app.post("/format_clinique/")
def format_clinique(request:QuestionRequest) -> dict:
    try:
        return dict(retrieval_format_clinique_lang(detect(request.question), model).invoke(request.question))
    except OutputParserException:
        try:
            return dict(retrieval_format_clinique_lang(detect(request.question), model).invoke(request.question))
        except OutputParserException:
            return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            return output_parser_exception
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                try:
                    return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
                except OutputParserException:
                    try:
                        return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
                    except Exception as e:
                        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                        return output_parser_exception
            case _ :
                try:
                    return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                            try:
                                return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
                            except OutputParserException:
                                try:
                                    return dict(retrieval_format_clinique_lang(detect(request.question), model_ggl).invoke(request.question))
                                except Exception as e:
                                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                                    return output_parser_exception
                        case _ :
                            return dict(retrieval_format_clinique_lang(detect(request.question), model).invoke(request.question))

@app.post("/summarize_consultation/")
def summarize_consultation(request:QuestionRequest) -> str:
    try:
        return retrieval_resume_consultation_lang(detect(request.question), model).invoke(request.question)
    except OutputParserException:
        try:
            return retrieval_resume_consultation_lang(detect(request.question), model).invoke(request.question)
        except OutputParserException:
            return retrieval_resume_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
        except Exception as e:
            rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
            return output_parser_exception
    except Exception as e:
        rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
        match e.status_code:
            case 400:
                return "Organization restricted"
            case 429:
                try:
                    return retrieval_resume_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                except OutputParserException:
                    return retrieval_resume_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    return output_parser_exception
            case _ :
                try:
                    return retrieval_resume_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                except Exception as e:
                    rich.print(f"\n\n ########## erreur #############\n\n {e}\n\n ########## methode #############\n\n {dir(e)}\n\n")
                    match e.status_code:
                        case 400:
                            return "Organization restricted"
                        case 429:
                             return retrieval_resume_consultation_lang(detect(request.question), model_ggl).invoke(request.question)
                        case _ :
                             return retrieval_resume_consultation_lang(detect(request.question), model).invoke(request.question)

# @app.post("/totalEnergieCongo")
# async def total_energie_congo(question:str, embedding)->str:
#     # load database
#     from qdrant_client import QdrantClient
#     from langchain_qdrant import QdrantVectorStore
#     client = QdrantClient(url="https://recette-apps.pategou.com:61268/")
#     # try:
#     #     print(client.get_collections())
#     # except Exception as e:
#     #     print(f"Erreur : {e}")
#     db = QdrantVectorStore(
#         client=client,
#         collection_name="TotalEnergieCongo",
#         embedding=embedding
#     )

#     retriever = db.as_retriever(
#         search_type="mmr", 
#             search_kwargs={
#                 "k": 10,
#                 "fetch_k":20,
#                 "lambda_mult":0.8,
#             }
#     ) 

#     from langchain_groq import ChatGroq
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import StrOutputParser
#     from langchain_core.runnables import RunnablePassthrough

#     template = """Répondez à la question en vous basant uniquement sur le contexte suivant:

#     {context}

#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#     model = ChatGroq(
#         model=os.getenv("GROQ_MODEL_NAME_2"),
#         temperature=.3,
#         api_key=os.getenv("GROQ_API_KEY")
#     )

    
#     def format_docs(docs):
#         return "\n\n".join([d.page_content for d in docs])


#     chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     return chain.invoke(question)

# @app.post("/transcribe-audio/")
# async def transcribe_audio(file: UploadFile = File(...)):
#     # print(f"start time : {time.time()}")
#     temp_audio_file = "temp_audio.mp3"
#     if os.path.exists(temp_audio_file):
#         # Supprimer le fichier temporaire après transcription
#         os.remove(temp_audio_file)
#     try:
#         # Vérifier que le fichier est bien un fichier audio
#         if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp3"]:
#             raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
 
#         # Charger le modèle Whisper
#         model = whisper.load_model("small")
#         import time
#         # Lire le fichier audio envoyé
#         print(f"Lecture time : {time.time()}")
#         audio = await file.read()

#         # import base64

#         # # ... code précédent ...

#         # with open(temp_audio_file, "rb") as temp_file:
#         #     audio_content = temp_file.read()
#         #     # Convertir les données audio en base64 pour l'affichage
#         #     encoded_audio = base64.b64encode(audio_content).decode()
#         #     print(f"Contenu du fichier audio (base64): {encoded_audio}")
            
#         # Sauvegarder temporairement le fichier audio
        
#         with open(temp_audio_file, "wb") as temp_file:
#             temp_file.write(audio)
 
#         # Transcrire le fichier audio avec Whisper
#         # print(f"start transcribe time : {time.time()}")
#         result = model.transcribe(temp_audio_file)
#         # print(f"end transcribe time : {time.time()}")
        
 
#         # Renvoyer le texte transcrit
#         print(f"\n\n {result["text"]}")
#         # print(f"\n\n\n end time : {time.time()}")
#         return {"text": result["text"]}
 
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")