"""
Parte 2 — Engenharia de Prompt: grafo LangGraph revisado.

Comparado com a Parte 1 (5 nós em sequência linear, 1 única chamada de
LLM), este grafo:

  - decompõe a responsabilidade única de "responder" em 3 chamadas de LLM
    especializadas (classificar, gerar, verificar), cada uma com seu
    próprio prompt em app/prompts.py (prompt decomposition);
  - encadeia a saída de uma etapa como entrada/condição da próxima
    (prompt chaining): a categoria decide se o fluxo segue para o RAG; a
    presença de evidência decide se a LLM de geração é chamada; o
    resultado da verificação decide se a resposta gerada é entregue;
  - evita chamar a LLM quando não há necessidade: pergunta fora do
    domínio ou saudação não geram busca nem chamada de geração; falta de
    evidência no índice vetorial (checada por distância, sem LLM) também
    não gera chamada de geração.

Nós:
    receive_question      — entrada da pergunta
    classify_question       — LLM #1: DOMINIO / FORA_DOMINIO / SAUDACAO
    respond_greeting          — resposta fixa, sem LLM (rota SAUDACAO)
    respond_out_of_domain       — resposta fixa, sem LLM (rota FORA_DOMINIO)
    retrieve_context               — busca no FAISS + checagem de evidência
                                       por distância (sem LLM)
    respond_no_evidence               — resposta fixa, sem LLM (rota sem evidência)
    generate_answer                     — LLM #2: gera a resposta usando o contexto
    verify_answer                         — LLM #3: confere se a resposta está
                                              sustentada pelo contexto
    format_response                        — retorno da resposta (anexa fontes
                                              apenas quando a resposta veio do RAG)
"""
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.config import EVIDENCE_DISTANCE_THRESHOLD, GENERATE_FEW_SHOT, GROQ_API_KEY, GROQ_MODEL
from app.graph_types import ChatMessage, GraphState
from app.retriever import retrieve_chunks_with_scores
import app.prompts as prompts

MAX_HISTORY_TURNS = 4  # quantas trocas (pergunta+resposta) anteriores entram no prompt de geração


# ---------------------------------------------------------------------------
# Instâncias da LLM, uma por responsabilidade, com temperatura adequada a
# cada tarefa: classificação e verificação são tarefas de decisão (baixa
# temperatura, mais determinístico); geração tem um pouco mais de liberdade
# para escrever uma frase natural.
# ---------------------------------------------------------------------------
_classify_llm = None
_generate_llm = None
_verify_llm = None


def _get_classify_llm() -> ChatGroq:
    global _classify_llm
    if _classify_llm is None:
        _classify_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    return _classify_llm


def _get_generate_llm() -> ChatGroq:
    global _generate_llm
    if _generate_llm is None:
        _generate_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)
    return _generate_llm


def _get_verify_llm() -> ChatGroq:
    global _verify_llm
    if _verify_llm is None:
        _verify_llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    return _verify_llm


# ---------------------------------------------------------------------------
# Nós
# ---------------------------------------------------------------------------

def receive_question_node(state: GraphState) -> GraphState:
    """Etapa 1: entrada da pergunta."""
    state["question"] = state["question"].strip()
    return state


def classify_question_node(state: GraphState) -> GraphState:
    """Etapa 2 (LLM #1): classifica a pergunta antes de qualquer busca."""
    llm = _get_classify_llm()
    messages = prompts.build_classify_messages(state["question"])
    response = llm.invoke(messages)
    result = prompts.parse_classify_output(response.content)
    state["category"] = result["categoria"]
    state["classify_confidence"] = result["confianca"]
    return state


def route_after_classify(state: GraphState) -> str:
    if state["category"] == "SAUDACAO":
        return "respond_greeting"
    if state["category"] == "FORA_DOMINIO":
        return "respond_out_of_domain"
    return "retrieve_context"


def respond_greeting_node(state: GraphState) -> GraphState:
    state["raw_answer"] = prompts.GREETING_ANSWER
    state["route_taken"] = "greeting"
    return state


def respond_out_of_domain_node(state: GraphState) -> GraphState:
    state["raw_answer"] = prompts.OUT_OF_DOMAIN_ANSWER
    state["route_taken"] = "out_of_domain"
    return state


def retrieve_context_node(state: GraphState) -> GraphState:
    """Etapa 3: recuperação de contexto + checagem de evidência (sem LLM).

    A checagem de evidência é feita por regra (distância L2 do FAISS), não
    por chamada de LLM — implementa literalmente a orientação da atividade
    de "utilizar o próprio fluxo do LangGraph para evitar a chamada da LLM
    quando não houver evidência suficiente, evitando consumo desnecessário
    de tokens".
    """
    results = retrieve_chunks_with_scores(state["question"])

    if not results:
        state["retrieved_context"] = ""
        state["sources"] = []
        state["has_evidence"] = False
        state["best_distance"] = None
        return state

    state["retrieved_context"] = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    state["sources"] = sorted({doc.metadata.get("source", "desconhecido") for doc, _ in results})

    best_distance = min(score for _, score in results)
    state["best_distance"] = best_distance
    state["has_evidence"] = best_distance <= EVIDENCE_DISTANCE_THRESHOLD

    return state


def route_after_retrieve(state: GraphState) -> str:
    return "generate_answer" if state["has_evidence"] else "respond_no_evidence"


def respond_no_evidence_node(state: GraphState) -> GraphState:
    state["raw_answer"] = prompts.NO_EVIDENCE_ANSWER
    state["route_taken"] = "no_evidence"
    return state


def generate_answer_node(state: GraphState) -> GraphState:
    """Etapa 4 (LLM #2): gera a resposta final usando o contexto recuperado."""
    llm = _get_generate_llm()
    messages = prompts.build_generate_answer_messages(
        context=state["retrieved_context"],
        question=state["question"],
        history=state["history"][-MAX_HISTORY_TURNS * 2:],
        few_shot=GENERATE_FEW_SHOT,
    )
    response = llm.invoke(messages)
    state["raw_answer"] = response.content
    state["route_taken"] = "generated"
    return state


def verify_answer_node(state: GraphState) -> GraphState:
    """Etapa 5 (LLM #3): confere se a resposta gerada é sustentada pelo contexto.

    Segunda camada de defesa: mesmo que o prompt de geração tenha sido
    manipulado por conteúdo malicioso embutido num chunk (indirect prompt
    injection), esta etapa roda de forma independente e pode reprovar a
    resposta antes que ela chegue ao usuário.
    """
    llm = _get_verify_llm()
    messages = prompts.build_verify_messages(state["retrieved_context"], state["raw_answer"])
    response = llm.invoke(messages)
    result = prompts.parse_verify_output(response.content)
    state["is_grounded"] = result["sustentada"]
    state["verify_justification"] = result["justificativa"]

    if not state["is_grounded"]:
        state["raw_answer"] = prompts.NOT_GROUNDED_ANSWER
        state["route_taken"] = "not_grounded"

    return state


def format_response_node(state: GraphState) -> GraphState:
    """Etapa final: retorno da resposta — anexa fontes apenas quando a
    resposta de fato veio do contexto recuperado (rota "generated")."""
    answer = state["raw_answer"]
    if state.get("route_taken") == "generated" and state.get("sources"):
        fontes = ", ".join(state["sources"])
        answer = f"{answer}\n\n_Fontes: {fontes}_"
    state["answer"] = answer
    return state


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("receive_question", receive_question_node)
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("respond_greeting", respond_greeting_node)
    graph.add_node("respond_out_of_domain", respond_out_of_domain_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("respond_no_evidence", respond_no_evidence_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("verify_answer", verify_answer_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("receive_question")
    graph.add_edge("receive_question", "classify_question")

    graph.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {
            "respond_greeting": "respond_greeting",
            "respond_out_of_domain": "respond_out_of_domain",
            "retrieve_context": "retrieve_context",
        },
    )

    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieve,
        {
            "generate_answer": "generate_answer",
            "respond_no_evidence": "respond_no_evidence",
        },
    )

    graph.add_edge("respond_greeting", "format_response")
    graph.add_edge("respond_out_of_domain", "format_response")
    graph.add_edge("respond_no_evidence", "format_response")
    graph.add_edge("generate_answer", "verify_answer")
    graph.add_edge("verify_answer", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Compila o grafo uma única vez e reaproveita entre requisições."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
