"""Testes do filtro de relevância da recuperação.

O índice FAISS real é gerado localmente por `python -m app.ingest` e não está
no repositório, então os testes trocam o vector store por um dublê que devolve
scores controlados. O que se verifica aqui é a regra de negócio do retriever
(quais chunks passam no corte), não a busca vetorial em si.
"""
import pytest
from langchain_core.documents import Document

from app import retriever


class FakeVectorStore:
    """Devolve sempre os mesmos pares (documento, score) da busca."""

    def __init__(self, results):
        self.results = results
        self.k_recebido = None

    def similarity_search_with_score(self, question, k):
        self.k_recebido = k
        return self.results


@pytest.fixture
def fake_store(monkeypatch):
    """Instala um vector store falso e devolve uma função para configurá-lo."""

    def instalar(results):
        store = FakeVectorStore(results)
        monkeypatch.setattr(retriever, "_get_vector_store", lambda: store)
        return store

    return instalar


def doc(texto, source="copa-do-mundo-fifa-de-1994"):
    return Document(page_content=texto, metadata={"source": source})


def test_mantem_apenas_chunks_dentro_do_limiar(fake_store):
    relevante = doc("O Brasil venceu a Copa de 1994.")
    irrelevante = doc("Texto sem relação com a pergunta.")
    fake_store([(relevante, 0.42), (irrelevante, 1.9)])

    resultado = retriever.retrieve_chunks("Quem venceu a Copa de 1994?")

    assert resultado == [relevante]


def test_score_exatamente_no_limiar_e_mantido(fake_store):
    """O corte é inclusivo (score <= limiar), então o limite exato passa."""
    limite = doc("Trecho no limite do corte.")
    fake_store([(limite, 1.0)])

    assert retriever.retrieve_chunks("pergunta") == [limite]


def test_sem_resultados_relevantes_devolve_lista_vazia(fake_store):
    """Sem contexto, o grafo monta um prompt vazio e a LLM responde que não
    encontrou a informação — em vez de inventar a resposta."""
    fake_store([(doc("Nada a ver."), 1.5), (doc("Também não."), 2.0)])

    assert retriever.retrieve_chunks("pergunta fora da base") == []


def test_preserva_a_ordem_de_relevancia_da_busca(fake_store):
    primeiro, segundo, terceiro = doc("a"), doc("b"), doc("c")
    fake_store([(primeiro, 0.1), (segundo, 0.5), (terceiro, 0.9)])

    assert retriever.retrieve_chunks("pergunta") == [primeiro, segundo, terceiro]


def test_usa_top_k_padrao_e_aceita_k_customizado(fake_store):
    store = fake_store([])
    retriever.retrieve_chunks("pergunta")
    assert store.k_recebido == retriever.TOP_K

    store = fake_store([])
    retriever.retrieve_chunks("pergunta", k=3)
    assert store.k_recebido == 3
