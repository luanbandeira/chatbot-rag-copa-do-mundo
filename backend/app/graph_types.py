"""
Tipos compartilhados entre graph.py, prompts.py e main.py.

Separado num módulo próprio (em vez de viver dentro de graph.py como na
Parte 1) para evitar import circular: prompts.py precisa do tipo
ChatMessage para formatar o histórico, e graph.py importa de prompts.py.
"""
from typing import List, Optional

# typing_extensions.TypedDict (não typing.TypedDict) porque o Pydantic
# (usado pelo FastAPI em main.py para validar o corpo da requisição) exige
# essa variante em Python < 3.12. Em Python 3.12+ os dois se comportam
# igual, mas assim o projeto roda sem esse erro em qualquer versão da
# equipe.
from typing_extensions import TypedDict


class ChatMessage(TypedDict):
    role: str  # "user" ou "assistant"
    content: str


class GraphState(TypedDict):
    question: str
    history: List[ChatMessage]

    # preenchido pelo nó classify_question
    category: str          # "DOMINIO" | "FORA_DOMINIO" | "SAUDACAO"
    classify_confidence: str

    # preenchido pelo nó retrieve_context
    retrieved_context: str
    sources: List[str]
    has_evidence: bool
    best_distance: Optional[float]

    # preenchido pelo nó generate_answer
    raw_answer: str

    # preenchido pelo nó verify_answer
    is_grounded: bool
    verify_justification: str

    # preenchido pelo nó format_response
    answer: str

    # rota final tomada pelo grafo, útil para os testes e para o relatório
    # antes/depois (ex.: "generated", "no_evidence", "out_of_domain")
    route_taken: str
