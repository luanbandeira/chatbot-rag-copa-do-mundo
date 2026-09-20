"""Testes dos nós do grafo LangGraph.

Nenhum teste aqui chama a API da Groq nem carrega o índice FAISS: a LLM e a
recuperação são substituídas por dublês, de modo que a suíte roda offline e
sem chave de API. O foco é a lógica de cada etapa do pipeline.
"""
from langchain_core.documents import Document

from app import graph


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Substitui ChatGroq: registra as chamadas e devolve uma resposta fixa."""

    def __init__(self, content="", erro=None):
        self.content = content
        self.erro = erro
        self.chamadas = []

    def __call__(self, **kwargs):
        """Permite usar a instância no lugar da classe ChatGroq."""
        return self

    def invoke(self, messages):
        self.chamadas.append(messages)
        if self.erro is not None:
            raise self.erro
        return FakeResponse(self.content)


def estado(**overrides):
    base = {
        "question": "Quem venceu a Copa de 1994?",
        "rewritten_question": "",
        "history": [],
        "retrieved_context": "",
        "sources": [],
        "prompt": "",
        "answer": "",
    }
    base.update(overrides)
    return base


def turno(role, content):
    return {"role": role, "content": content}


# --- Etapa 1: entrada da pergunta -------------------------------------------


def test_receive_question_remove_espacos_das_pontas():
    resultado = graph.receive_question_node(estado(question="  Quem venceu?   "))

    assert resultado["question"] == "Quem venceu?"


# --- Etapa 2: contextualização da pergunta ----------------------------------


def test_rewrite_sem_historico_nao_chama_a_llm(monkeypatch):
    """Sem histórico não há o que contextualizar; poupar a chamada evita
    latência e consumo de cota à toa."""
    llm = FakeLLM(content="nao deveria ser usado")
    monkeypatch.setattr(graph, "ChatGroq", llm)

    resultado = graph.rewrite_question_node(estado(question="Quem venceu em 1994?"))

    assert resultado["rewritten_question"] == "Quem venceu em 1994?"
    assert llm.chamadas == []


def test_rewrite_com_historico_usa_a_pergunta_reescrita(monkeypatch):
    llm = FakeLLM(content="Quem venceu a Copa do Mundo de 2002?")
    monkeypatch.setattr(graph, "ChatGroq", llm)

    resultado = graph.rewrite_question_node(estado(
        question="E em 2002?",
        history=[
            turno("user", "Quem venceu a Copa de 1994?"),
            turno("assistant", "O Brasil."),
        ],
    ))

    assert resultado["rewritten_question"] == "Quem venceu a Copa do Mundo de 2002?"
    assert len(llm.chamadas) == 1


def test_rewrite_envia_o_historico_no_prompt(monkeypatch):
    llm = FakeLLM(content="pergunta reescrita")
    monkeypatch.setattr(graph, "ChatGroq", llm)

    graph.rewrite_question_node(estado(
        question="E em 2002?",
        history=[turno("user", "Quem venceu a Copa de 1994?")],
    ))

    prompt = llm.chamadas[0][0].content
    assert "Quem venceu a Copa de 1994?" in prompt
    assert "E em 2002?" in prompt


def test_rewrite_cai_para_a_pergunta_original_se_a_llm_falhar(monkeypatch):
    """Uma falha na contextualização não pode derrubar a resposta: o pipeline
    segue com a pergunta como o usuário escreveu."""
    monkeypatch.setattr(graph, "ChatGroq", FakeLLM(erro=RuntimeError("API fora do ar")))

    resultado = graph.rewrite_question_node(estado(
        question="E em 2002?",
        history=[turno("user", "Quem venceu a Copa de 1994?")],
    ))

    assert resultado["rewritten_question"] == "E em 2002?"


def test_rewrite_cai_para_a_pergunta_original_se_a_llm_responder_vazio(monkeypatch):
    monkeypatch.setattr(graph, "ChatGroq", FakeLLM(content="   "))

    resultado = graph.rewrite_question_node(estado(
        question="E em 2002?",
        history=[turno("user", "Quem venceu a Copa de 1994?")],
    ))

    assert resultado["rewritten_question"] == "E em 2002?"


# --- Etapa 3: recuperação de contexto ---------------------------------------


def test_retrieve_usa_a_pergunta_reescrita_na_busca(monkeypatch):
    perguntas = []

    def fake_retrieve(pergunta):
        perguntas.append(pergunta)
        return []

    monkeypatch.setattr(graph, "retrieve_chunks", fake_retrieve)

    graph.retrieve_context_node(estado(
        question="E em 2002?",
        rewritten_question="Quem venceu a Copa do Mundo de 2002?",
    ))

    assert perguntas == ["Quem venceu a Copa do Mundo de 2002?"]


def test_retrieve_lista_fontes_por_titulo_sem_repetir(monkeypatch):
    chunks = [
        Document(page_content="a", metadata={"title": "Copa do mundo FIFA de 1994"}),
        Document(page_content="b", metadata={"title": "Copa do mundo FIFA de 1994"}),
        Document(page_content="c", metadata={"title": "Copa do mundo FIFA"}),
    ]
    monkeypatch.setattr(graph, "retrieve_chunks", lambda pergunta: chunks)

    resultado = graph.retrieve_context_node(estado())

    assert resultado["sources"] == ["Copa do mundo FIFA", "Copa do mundo FIFA de 1994"]


def test_retrieve_usa_o_slug_quando_o_chunk_nao_tem_titulo(monkeypatch):
    """Chunks indexados antes de o título passar a ser gravado nos metadados."""
    chunks = [Document(page_content="a", metadata={"source": "copa-do-mundo-fifa-de-1994"})]
    monkeypatch.setattr(graph, "retrieve_chunks", lambda pergunta: chunks)

    resultado = graph.retrieve_context_node(estado())

    assert resultado["sources"] == ["copa-do-mundo-fifa-de-1994"]


def test_retrieve_junta_o_conteudo_dos_chunks_no_contexto(monkeypatch):
    chunks = [
        Document(page_content="Primeiro trecho.", metadata={"title": "A"}),
        Document(page_content="Segundo trecho.", metadata={"title": "B"}),
    ]
    monkeypatch.setattr(graph, "retrieve_chunks", lambda pergunta: chunks)

    resultado = graph.retrieve_context_node(estado())

    assert "Primeiro trecho." in resultado["retrieved_context"]
    assert "Segundo trecho." in resultado["retrieved_context"]


def test_retrieve_sem_chunks_deixa_contexto_e_fontes_vazios(monkeypatch):
    monkeypatch.setattr(graph, "retrieve_chunks", lambda pergunta: [])

    resultado = graph.retrieve_context_node(estado())

    assert resultado["retrieved_context"] == ""
    assert resultado["sources"] == []


# --- Etapa 4: montagem do prompt --------------------------------------------


def test_build_prompt_reune_contexto_historico_e_pergunta():
    resultado = graph.build_prompt_node(estado(
        question="E em 2002?",
        retrieved_context="O Brasil venceu em 2002.",
        history=[
            turno("user", "Quem venceu a Copa de 1994?"),
            turno("assistant", "O Brasil venceu em 1994."),
        ],
    ))

    prompt = resultado["prompt"]
    assert "O Brasil venceu em 2002." in prompt
    assert "Usuário: Quem venceu a Copa de 1994?" in prompt
    assert "Assistente: O Brasil venceu em 1994." in prompt
    assert "PERGUNTA ATUAL: E em 2002?" in prompt


def test_build_prompt_sinaliza_ausencia_de_historico():
    resultado = graph.build_prompt_node(estado(history=[]))

    assert "(sem histórico anterior)" in resultado["prompt"]


def test_build_prompt_limita_o_historico_as_ultimas_trocas():
    """Histórico longo estoura a janela de contexto, então só as últimas
    MAX_HISTORY_TURNS trocas entram no prompt."""
    history = []
    for i in range(graph.MAX_HISTORY_TURNS + 3):
        history.append(turno("user", f"pergunta {i}"))
        history.append(turno("assistant", f"resposta {i}"))

    resultado = graph.build_prompt_node(estado(history=history))

    assert "pergunta 0" not in resultado["prompt"]
    assert f"pergunta {graph.MAX_HISTORY_TURNS + 2}" in resultado["prompt"]


# --- Etapa 6: formatação da resposta ----------------------------------------


def test_format_response_nao_concatena_as_fontes_no_texto():
    """As fontes vão para o frontend no campo `sources`; repeti-las no texto
    faria esse rodapé voltar para a LLM dentro do histórico."""
    resultado = graph.format_response_node(estado(
        answer="O Brasil venceu em 1994.",
        sources=["Copa do mundo FIFA de 1994"],
    ))

    assert resultado["answer"] == "O Brasil venceu em 1994."
    assert resultado["sources"] == ["Copa do mundo FIFA de 1994"]


def test_format_response_remove_espacos_das_pontas():
    resultado = graph.format_response_node(estado(answer="   O Brasil venceu.   "))

    assert resultado["answer"] == "O Brasil venceu."
