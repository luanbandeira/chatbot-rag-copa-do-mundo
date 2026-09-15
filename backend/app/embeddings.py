"""
Wrapper do modelo de embeddings local (sentence-transformers) implementando
a interface Embeddings do LangChain, para poder ser usado tanto na
indexação (ingest.py) quanto na busca em tempo real (retriever.py).

Modelo: intfloat/multilingual-e5-small — multilíngue, leve (~470MB), roda
em CPU e foi treinado para recuperação (pergunta -> trecho que responde),
diferente dos modelos "paraphrase-*", que só comparam frases parecidas.
Aceita até 512 tokens por texto.

O E5 exige os prefixos "query: " (perguntas) e "passage: " (trechos da base).
Os vetores saem normalizados, então a distância L2 padrão do FAISS ordena
os resultados do mesmo jeito que a similaridade de cosseno.
"""
from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

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
        vectors = model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = _get_model()
        vector = model.encode([f"query: {text}"], convert_to_numpy=True, normalize_embeddings=True)
        return vector[0].tolist()


def get_embeddings() -> LocalEmbeddings:
    return LocalEmbeddings()