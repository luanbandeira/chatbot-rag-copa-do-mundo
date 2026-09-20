"""
Grafo LangGraph que orquestra o fluxo de RAG do chatbot.

Nós, mapeando diretamente para as etapas centrais exigidas pela atividade:
    1. receive_question   — entrada da pergunta
    2. rewrite_question   — contextualização da pergunta usando o histórico
    3. retrieve_context   — recuperação de contexto
    4. build_prompt       — montagem do prompt (contexto + histórico)
    5. call_llm           — chamada da LLM externa (Groq)
    6. format_response    — retorno da resposta
"""
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.retriever import retrieve_chunks

MAX_HISTORY_TURNS = 4  # quantas trocas (pergunta+resposta) anteriores entram no prompt

SYSTEM_PROMPT = (
    "Você é um assistente especializado em Copa do Mundo FIFA. "
    "Responda SOMENTE com base no CONTEXTO fornecido abaixo, extraído de uma "
    "base de conhecimento sobre a Copa do Mundo. Se a resposta não estiver "
    "no contexto, diga claramente que não encontrou essa informação na base "
    "de conhecimento — não invente dados. Responda em português, de forma "
    "clara e objetiva."
)


class ChatMessage(TypedDict):
    role: str  # "user" ou "assistant"
    content: str


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    history: List[ChatMessage]
    retrieved_context: str
    sources: List[str]
    prompt: str
    answer: str


def receive_question_node(state: GraphState) -> GraphState:
    """Etapa 1: entrada da pergunta.

    A pergunta já chega validada pelo endpoint FastAPI; esse nó existe
    para deixar essa etapa explícita no grafo, conforme pedido pela
    atividade.
    """
    state["question"] = state["question"].strip()
    return state

def rewrite_question_node(state: GraphState) -> GraphState:
    """Etapa 2: contextualização da pergunta usando o histórico."""

    history = state.get("history", [])

    # Se não houver histórico, não é necessário chamar a LLM.
    if not history:
        state["rewritten_question"] = state["question"]
        return state

    recent_history = history[-MAX_HISTORY_TURNS * 2:]

    history_text = ""

    for turn in recent_history:
        prefixo = "Usuário" if turn["role"] == "user" else "Assistente"
        history_text += f"{prefixo}: {turn['content']}\n"

    rewrite_prompt = f"""
Você é responsável por preparar perguntas para um sistema de busca
sobre a Copa do Mundo FIFA.

Use o histórico da conversa para transformar a PERGUNTA ATUAL em uma
pergunta completa e independente.

REGRAS:
- Preserve exatamente a intenção do usuário.
- Use o histórico somente quando ele for necessário para entender a pergunta.
- Não responda à pergunta.
- Não invente informações.
- Retorne SOMENTE a pergunta reescrita.
- Se a pergunta atual já for independente e clara, apenas repita-a.

HISTÓRICO:
{history_text}

PERGUNTA ATUAL:
{state["question"]}

PERGUNTA REESCRITA:
""".strip()

    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0
        )

        response = llm.invoke([
            HumanMessage(content=rewrite_prompt)
        ])

        rewritten = response.content.strip()

        print(f"[RAG] Pergunta original: {state['question']}")
        print(f"[RAG] Pergunta reescrita: {rewritten}")

        # Fallback de segurança
        if rewritten:
            state["rewritten_question"] = rewritten
        else:
            state["rewritten_question"] = state["question"]

    except Exception as e:
        print(f"[rewrite_question] Erro ao reescrever pergunta: {e}")
        state["rewritten_question"] = state["question"]

    return state


def retrieve_context_node(state: GraphState) -> GraphState:
    """Etapa 3: recuperação dos chunks mais relevantes da base vetorial."""
    docs = retrieve_chunks(state["rewritten_question"])
    state["retrieved_context"] = "\n\n---\n\n".join(doc.page_content for doc in docs)
    # O título ("Copa do mundo FIFA de 2022") é mais legível para quem usa o
    # chat do que o slug do arquivo; o slug fica como reserva.
    state["sources"] = sorted({
        doc.metadata.get("title") or doc.metadata.get("source", "desconhecido")
        for doc in docs
    })
    return state


def build_prompt_node(state: GraphState) -> GraphState:
    """Etapa 4: montagem do prompt, combinando contexto recuperado e histórico."""
    history_text = ""
    recent_history = state["history"][-MAX_HISTORY_TURNS * 2:]
    for turn in recent_history:
        prefixo = "Usuário" if turn["role"] == "user" else "Assistente"
        history_text += f"{prefixo}: {turn['content']}\n"

    state["prompt"] = (
        f"CONTEXTO:\n{state['retrieved_context']}\n\n"
        f"HISTÓRICO DA CONVERSA:\n{history_text or '(sem histórico anterior)'}\n\n"
        f"PERGUNTA ATUAL: {state['question']}"
    )
    return state


def call_llm_node(state: GraphState) -> GraphState:
    """Etapa 5: chamada à LLM externa (Groq) para gerar a resposta final."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["prompt"]),
    ]
    response = llm.invoke(messages)
    state["answer"] = response.content
    return state


def format_response_node(state: GraphState) -> GraphState:
    """Etapa 6: retorno da resposta.

    As fontes seguem como campo próprio do estado (e da resposta da API) em
    vez de serem concatenadas no texto. Assim o frontend pode exibi-las como
    elementos separados e o histórico enviado à LLM não carrega esse rodapé.
    """
    state["answer"] = state["answer"].strip()
    return state


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("receive_question", receive_question_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("receive_question")

    graph.add_edge("receive_question", "rewrite_question")
    graph.add_edge("rewrite_question", "retrieve_context")
    graph.add_edge("retrieve_context", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Compila o grafo uma única vez e reaproveita entre requisições."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
