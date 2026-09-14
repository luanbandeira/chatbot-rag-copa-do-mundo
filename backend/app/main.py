"""
Ponto de entrada da API FastAPI.
Por enquanto só expõe um health check — o endpoint /chat será
adicionado quando o grafo LangGraph (RAG) estiver implementado.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chatbot RAG - Copa do Mundo")

# Libera acesso do frontend Next.js (rodando em localhost:3000) durante o desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
