"""
Carrega o índice FAISS salvo em disco (gerado por ingest.py) e expõe uma
função de busca por similaridade, usada pelo nó de recuperação do grafo
LangGraph.
"""
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import FAISS_INDEX_DIR
from app.embeddings import get_embeddings

TOP_K = 8  # quantos chunks recuperar por pergunta

_vector_store = None  # cache simples para não recarregar o índice a cada chamada


def _get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        embeddings = get_embeddings()
        _vector_store = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,  # seguro aqui: o índice é gerado por nós mesmos
        )
    return _vector_store


def retrieve_chunks(question: str, k: int = TOP_K) -> List[Document]:
    """Retorna os k chunks mais relevantes semanticamente para a pergunta."""
    vector_store = _get_vector_store()
    return vector_store.similarity_search(question, k=k)