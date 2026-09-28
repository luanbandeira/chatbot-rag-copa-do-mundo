"""
Ponto de entrada da API FastAPI.
Expõe o endpoint /chat, que dispara o grafo LangGraph de RAG.

Parte 2: a resposta agora também informa route_taken e category — não
exigido pela interface (que só lê .answer), mas útil para o conjunto de
testes (testes/rodar_testes.py) inspecionar qual caminho do grafo foi
percorrido sem precisar reler o log do servidor.
"""
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph_types import ChatMessage
from app.graph import get_graph

app = FastAPI(title="Chatbot RAG - Copa do Mundo")

# Libera acesso do frontend Next.js (rodando em localhost:3000) durante o desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    category: Optional[str] = None
    route_taken: Optional[str] = None
    best_distance: Optional[float] = None


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
    return ChatResponse(
        answer=result["answer"],
        category=result.get("category"),
        route_taken=result.get("route_taken"),
        best_distance=result.get("best_distance"),
    )
