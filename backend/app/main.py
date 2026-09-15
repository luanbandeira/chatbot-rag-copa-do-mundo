"""
Ponto de entrada da API FastAPI.
Expõe o endpoint /chat, que dispara o grafo LangGraph de RAG.
"""
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph import ChatMessage, get_graph

app = FastAPI(title="Chatbot RAG - Copa do Mundo")

# Libera acesso do frontend Next.js rodando localmente em qualquer porta
# (se a 3000 estiver ocupada, o Next sobe na 3001, 3002...)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    graph = get_graph()
    result = graph.invoke({
        "question": request.question,
        "history": request.history,
    })
    return ChatResponse(answer=result["answer"])