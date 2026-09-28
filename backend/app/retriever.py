"""
Carrega o índice FAISS salvo em disco (gerado por ingest.py) e expõe uma
função de busca por similaridade, usada pelo nó de recuperação do grafo
LangGraph.

Parte 2 (Engenharia de Prompt): retrieve_chunks_with_scores() também
devolve a distância L2 de cada chunk, usada pelo nó check_evidence do
grafo para decidir se existe evidência suficiente ANTES de gastar uma
chamada de LLM — evita consumo desnecessário de tokens quando a pergunta
não tem nenhum chunk realmente relevante na base.
"""
from typing import List, Tuple

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


def retrieve_chunks_with_scores(question: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    """Retorna os k chunks mais relevantes junto com a distância L2 de cada um.

    Distância L2 menor = mais similar. Usado para decidir se há evidência
    suficiente sem precisar chamar a LLM (ver app/graph.py, check_evidence_node).
    """
    vector_store = _get_vector_store()
    return vector_store.similarity_search_with_score(question, k=k)
