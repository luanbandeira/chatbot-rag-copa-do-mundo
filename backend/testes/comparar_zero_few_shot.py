"""Compara o prompt de geração em zero-shot e em few-shot (atividade, item 5).

Como o experimento é montado
----------------------------
O objetivo é isolar UMA variável: a presença dos exemplos em
`prompts.GENERATE_ANSWER_EXAMPLES`. Por isso o script:

  1. recupera o contexto do FAISS UMA única vez por pergunta;
  2. monta as mensagens duas vezes a partir do mesmo contexto, via
     `prompts.build_generate_answer_messages(..., few_shot=False)` e
     `(..., few_shot=True)`;
  3. chama a LLM de geração nas duas versões, com a mesma temperatura.

Isso é mais controlado do que subir o servidor duas vezes trocando
`GENERATE_FEW_SHOT` no `.env`: aqui as duas variantes recebem exatamente o
mesmo contexto recuperado, então qualquer diferença na resposta vem do
prompt, não de variação na recuperação.

Em produção, quem decide a variante continua sendo a variável de ambiente
`GENERATE_FEW_SHOT` (ver `app/config.py`), lida por `app/graph.py`. Este
script não altera nem lê esse toggle — ele exercita as duas variantes
diretamente.

Como rodar (de dentro de `backend/`, com o venv ativado e o `.env`
preenchido com GROQ_API_KEY):

    python -m testes.comparar_zero_few_shot

Resultados: `docs/dados/zero-vs-few-shot.json` (respostas completas) e um
resumo impresso no terminal.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

# Permite rodar como `python testes/comparar_zero_few_shot.py` também.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# O console do Windows usa cp1252 e quebra ao imprimir caracteres que a LLM
# devolve de vez em quando (hifen nao separavel, aspas tipograficas). Os
# arquivos de resultado ja sao gravados em UTF-8; isso conserta so o terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.retriever import retrieve_chunks_with_scores
import app.prompts as prompts

# Mesma temperatura usada pelo nó de geração em app/graph.py, para que a
# comparação reflita o comportamento real da aplicação.
TEMPERATURA_GERACAO = 0.2

SAIDA = Path(__file__).resolve().parent.parent.parent / "docs" / "dados" / "zero-vs-few-shot.json"


# Cada caso declara o que se espera observar, para a análise não virar
# impressão vaga depois. "abstencao" = a resposta correta é dizer que não
# encontrou a informação na base.
CASOS = [
    {
        "id": "factual-simples",
        "pergunta": "Quem venceu a Copa do Mundo de 1970?",
        "esperado": "Resposta curta citando o Brasil, sustentada pelo contexto.",
        "deve_abster": False,
    },
    {
        "id": "factual-com-detalhe",
        "pergunta": "Onde foi disputada a final da Copa do Mundo de 2014 e quem ganhou?",
        "esperado": "Cita Alemanha campeã; local sustentado pelo contexto.",
        "deve_abster": False,
    },
    {
        "id": "detalhe-ausente",
        "pergunta": "Quantos espectadores assistiram à final da Copa do Mundo de 1950 no Maracanã, com número exato?",
        "esperado": "Abstenção OU número explicitamente presente no contexto; não pode inventar.",
        "deve_abster": None,  # depende do que a base cobre; avaliado manualmente
    },
    {
        "id": "fora-da-base",
        "pergunta": "Qual foi o placar da final da Liga dos Campeões de 2019?",
        "esperado": "Abstenção — assunto fora da base de Copa do Mundo.",
        "deve_abster": True,
    },
    {
        "id": "pede-resposta-longa",
        "pergunta": "Faça um resumo completo e bem detalhado, com todos os detalhes possíveis, da história inteira das Copas do Mundo.",
        "esperado": "Deve respeitar o limite de 3 frases do prompt mesmo com o pedido de detalhamento.",
        "deve_abster": False,
    },
    {
        "id": "conhecimento-proprio",
        "pergunta": "Quantos títulos mundiais o Brasil tem e em que anos foram conquistados?",
        "esperado": "Só pode afirmar o que estiver no contexto; não pode completar de memória.",
        "deve_abster": None,
    },
]


def contar_frases(texto: str) -> int:
    """Conta frases de forma aproximada, para medir aderência ao limite de 3."""
    partes = [p for p in re.split(r"[.!?]+(?:\s|$)", texto.strip()) if p.strip()]
    return len(partes)


def parece_abstencao(texto: str) -> bool:
    """Detecta se a resposta é uma recusa por falta de evidência."""
    normalizado = texto.lower()
    marcadores = [
        "não encontrei",
        "nao encontrei",
        "não foi encontrada",
        "nao foi encontrada",
        "não está disponível",
        "nao esta disponivel",
        "não consta",
        "nao consta",
    ]
    return any(m in normalizado for m in marcadores)


# O plano gratuito da Groq limita tokens por minuto (TPM). Cada chamada aqui
# manda o contexto inteiro (8 chunks), então duas chamadas seguidas estouram o
# limite. Esperar entre as chamadas e respeitar o tempo sugerido pelo erro 429
# é o que mantém a suíte rodando até o fim sem perder casos.
PAUSA_ENTRE_CHAMADAS = 22  # segundos
MAX_TENTATIVAS = 5


def _segundos_sugeridos(erro: Exception, padrao: float) -> float:
    """Extrai o 'try again in 12.84s' da mensagem de rate limit da Groq."""
    achado = re.search(r"try again in ([0-9.]+)s", str(erro))
    return float(achado.group(1)) + 2 if achado else padrao


def _invocar(llm, mensagens) -> str:
    """Chama a LLM tolerando rate limit e resposta vazia.

    Resposta vazia é tratada como falha e não como resultado: registrar ""
    como se fosse a resposta do modelo falsearia a comparação.
    """
    espera = 10.0
    ultimo_erro = None

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            texto = llm.invoke(mensagens).content.strip()
            if texto:
                return texto
            ultimo_erro = RuntimeError("a LLM devolveu resposta vazia")
            print(f"    (resposta vazia, tentativa {tentativa}/{MAX_TENTATIVAS})")
            time.sleep(espera)
        except Exception as erro:  # inclui groq.RateLimitError
            ultimo_erro = erro
            if tentativa == MAX_TENTATIVAS:
                break
            pausa = _segundos_sugeridos(erro, espera)
            print(f"    (falha: {type(erro).__name__}; aguardando {pausa:.0f}s "
                  f"e tentando de novo — {tentativa}/{MAX_TENTATIVAS})")
            time.sleep(pausa)
            espera = min(espera * 2, 60)

    raise RuntimeError(f"falhou após {MAX_TENTATIVAS} tentativas: {ultimo_erro}")


def salvar(registros: list) -> None:
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(
            {"modelo": GROQ_MODEL, "temperatura": TEMPERATURA_GERACAO, "casos": registros},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def gerar(llm, contexto: str, pergunta: str, few_shot: bool) -> str:
    mensagens = prompts.build_generate_answer_messages(
        context=contexto,
        question=pergunta,
        history=[],  # histórico vazio nos dois lados: mantém a comparação controlada
        few_shot=few_shot,
    )
    return _invocar(llm, mensagens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara o prompt de geracao em zero-shot e few-shot.")
    parser.add_argument("--repeticoes", type=int, default=1,
                        help="quantas vezes repetir cada caso em cada variante "
                             "(>1 mede a estabilidade da resposta entre execucoes)")
    parser.add_argument("--casos", default="",
                        help="lista de ids separados por virgula para rodar so um subconjunto")
    args = parser.parse_args()

    selecionados = [c.strip() for c in args.casos.split(",") if c.strip()]
    casos = [c for c in CASOS if not selecionados or c["id"] in selecionados]

    if not GROQ_API_KEY:
        print("ERRO: GROQ_API_KEY não configurada. Copie .env.example para .env "
              "e preencha sua chave antes de rodar.")
        return 1

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=TEMPERATURA_GERACAO)

    registros = []

    for caso in casos:
        pergunta = caso["pergunta"]
        print(f"\n{'=' * 78}\n[{caso['id']}] {pergunta}")

        resultados = retrieve_chunks_with_scores(pergunta)
        contexto = "\n\n---\n\n".join(doc.page_content for doc, _ in resultados)
        # O FAISS devolve numpy.float32, que o json não serializa.
        melhor_distancia = min((float(score) for _, score in resultados), default=None)
        fontes = sorted({doc.metadata.get("source", "desconhecido") for doc, _ in resultados})

        zeros, fews = [], []
        for _ in range(args.repeticoes):
            zeros.append(gerar(llm, contexto, pergunta, few_shot=False))
            time.sleep(PAUSA_ENTRE_CHAMADAS)
            fews.append(gerar(llm, contexto, pergunta, few_shot=True))
            time.sleep(PAUSA_ENTRE_CHAMADAS)

        zero, few = zeros[0], fews[0]

        registro = {
            "id": caso["id"],
            "pergunta": pergunta,
            "esperado": caso["esperado"],
            "deve_abster": caso["deve_abster"],
            "melhor_distancia": melhor_distancia,
            "fontes": fontes,
            "repeticoes": args.repeticoes,
            "zero_shot": {
                "resposta": zero,
                "todas": zeros,
                "distintas": len(set(zeros)),
                "frases": contar_frases(zero),
                "caracteres": len(zero),
                "abstencao": parece_abstencao(zero),
            },
            "few_shot": {
                "resposta": few,
                "todas": fews,
                "distintas": len(set(fews)),
                "frases": contar_frases(few),
                "caracteres": len(few),
                "abstencao": parece_abstencao(few),
            },
        }
        registros.append(registro)

        print(f"  distancia={melhor_distancia:.4f}")
        print(f"  ZERO-SHOT ({registro['zero_shot']['frases']} frases, "
              f"{registro['zero_shot']['caracteres']} chars): {zero}")
        print(f"  FEW-SHOT  ({registro['few_shot']['frases']} frases, "
              f"{registro['few_shot']['caracteres']} chars): {few}")

        # Salva a cada caso: a corrida inteira leva minutos por causa do rate
        # limit, e uma falha no último caso não pode custar os anteriores.
        salvar(registros)

    print(f"\n{'=' * 78}")
    print(f"Resultados salvos em {SAIDA}")

    if args.repeticoes > 1:
        print(f"\nEstabilidade (respostas distintas em "
              f"{args.repeticoes} execuções do mesmo caso):")
        for r in registros:
            print(f"  {r['id']:<22} zero-shot: {r['zero_shot']['distintas']}"
                  f"  |  few-shot: {r['few_shot']['distintas']}")

    fora_do_limite_zero = sum(1 for r in registros if r["zero_shot"]["frases"] > 3)
    fora_do_limite_few = sum(1 for r in registros if r["few_shot"]["frases"] > 3)
    print(f"Respostas acima de 3 frases — zero-shot: {fora_do_limite_zero}/{len(registros)}, "
          f"few-shot: {fora_do_limite_few}/{len(registros)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
