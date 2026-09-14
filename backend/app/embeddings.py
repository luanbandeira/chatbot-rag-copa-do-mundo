"""
Wrapper do modelo de embeddings local (sentence-transformers) implementando
a interface Embeddings do LangChain, para poder ser usado tanto na
indexação (ingest.py) quanto na busca em tempo real (retriever.py).

Modelo: paraphrase-multilingual-MiniLM-L12-v2 — multilíngue, leve
(~470MB), funciona bem em português e roda em CPU sem problema.
"""
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None  # cache simples para não recarregar o modelo a cada chamada


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Carregando modelo de embeddings '{MODEL_NAME}' "
              f"(primeira vez baixa ~470MB, depois fica em cache local)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class LocalEmbeddings(Embeddings):
    """Implementação da interface Embeddings do LangChain usando sentence-transformers local."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        vector = model.encode([text], convert_to_numpy=True)
        return vector[0].tolist()


def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()