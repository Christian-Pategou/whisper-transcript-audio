from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agent import define_graph
from typing_extensions import Union, Dict
from scheduler import start_scheduler
import db

class Request(BaseModel):
    question: str
    id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage de l'application
    start_scheduler()
    print("[INFO] Scheduler démarré.")

    yield

    # Code exécuté à l'arrêt de l'application (facultatif)
    print("[INFO] Application arrêtée.")


app = FastAPI(lifespan=lifespan)

# Ajoute le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet uniquement les origines spÃ©cifiÃ©es
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les mÃ©thodes HTTP (GET, POST, etc.)
    allow_headers=["*"],   # Permet tous les en-tÃªtes
)


@app.get("/support-links")
async def get_support_links():
    return {"context": db.support_links_context}


graph = define_graph()
async def stream_response(request: Request):
    config = {
        "configurable": {
            "thread_id": request.id
        }
    }
    inputs = {
        "question": request.question,
        "max_iter": 0
    }

    async for event in graph.astream(input=inputs, config=config, stream_mode="messages"):
        yield event[0].content

@app.post("/support-stream/")
async def generate_response(request:Request) -> str:
    try:
        return StreamingResponse(stream_response(request=request), media_type="text/event-stream") #event-stream
    except Exception as e:
        print(f"an error is occured: {e}")

@app.post("/support")
async def generate(request:Request) -> Union[Dict, str]:
    config = {
        "configurable": {
            "thread_id": request.id
        }
    }
    inputs = {
        "question": request.question,
        "max_iter": 0
    }
    res = await graph.ainvoke(input=inputs, config=config,)
    return res.get("answer", "Quelque chose à mal fonctionner, désolé.")




