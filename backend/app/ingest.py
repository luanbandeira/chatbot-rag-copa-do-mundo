"""
Pipeline de ingestão do RAG: lê os textos brutos coletados em data/raw/,
limpa, quebra em chunks, gera embeddings e monta o índice vetorial FAISS.

Uso:
    python -m app.ingest
"""
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.config import FAISS_INDEX_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from app.embeddings import get_embeddings

# Seções de baixo valor para RAG que a Wikipédia costuma deixar no fim dos artigos
NOISE_SECTION_HEADERS = {
    "Ver também",
    "Referências",
    "Ligações externas",
    "Bibliografia",
    "Notas",
}

MIN_CLEANED_LENGTH = 500  # documentos mais curtos que isso após limpeza são descartados
MIN_CHUNK_LENGTH = 80  # chunks menores que isso são só restos de título/tabela, não ajudam na busca

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

SECTION_HEADER_RE = re.compile(r"^\s*=+\s*.*?\s*=+\s*$")  # ex: "=== Final ==="


def remove_empty_sections(lines: List[str]) -> List[str]:
    """Remove cabeçalhos de seção sem conteúdo.

    O texto puro da Wikipédia perde as tabelas (chaveamento, placares,
    artilharia), deixando sequências de cabeçalhos vazios como
    "=== Semifinais ===  === Final ===". Esses trechos viravam chunks sem
    informação que apareciam no topo da busca.
    """
    kept = []
    for i, line in enumerate(lines):
        if SECTION_HEADER_RE.match(line):
            next_line = next((l for l in lines[i + 1:] if l.strip()), None)
            if next_line is None or SECTION_HEADER_RE.match(next_line):
                continue
        kept.append(line)
    return kept


def title_from_source(source: str) -> str:
    """'copa-do-mundo-fifa-de-2022' -> 'Copa do mundo FIFA de 2022'."""
    title = source.replace("-", " ").replace("fifa", "FIFA")
    return title[:1].upper() + title[1:]


def clean_text(text: str) -> str:
    """Remove seções de baixo valor (referências, links externos etc.).

    Essas seções sempre aparecem no final do artigo na Wikipédia, então
    basta cortar tudo a partir da primeira ocorrência de qualquer uma delas.
    """
    lines = text.split("\n")
    truncate_at = None

    for i, line in enumerate(lines):
        if line.strip() in NOISE_SECTION_HEADERS:
            truncate_at = i
            break

    if truncate_at is not None:
        lines = lines[:truncate_at]

    text = "\n".join(remove_empty_sections(lines))
    text = re.sub(r"\n{3,}", "\n\n", text)  # remove excesso de linhas em branco
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_and_clean_documents() -> List[Document]:
    raw_dir = Path(RAW_DATA_DIR)
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for path in sorted(raw_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw_text)

        if len(cleaned) < MIN_CLEANED_LENGTH:
            print(f"  [PULADO] {path.name} — só {len(cleaned)} caracteres após limpeza (provável tabela/lista)")
            continue

        documents.append(Document(page_content=cleaned, metadata={"source": path.stem}))
        (processed_dir / path.name).write_text(cleaned, encoding="utf-8")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for chunk in splitter.split_documents(documents):
        if len(chunk.page_content.strip()) < MIN_CHUNK_LENGTH:
            continue
        # Um trecho no meio do artigo muitas vezes não cita o ano da edição
        # ("a Argentina conquistou seu terceiro título..."). Prefixar o título
        # do artigo liga o chunk à edição certa, tanto na busca quanto para a LLM.
        title = title_from_source(chunk.metadata["source"])
        chunk.page_content = f"{title}: {chunk.page_content}"
        chunk.metadata["title"] = title
        chunks.append(chunk)
    return chunks


def build_and_save_index(chunks: List[Document]) -> None:
    embeddings = get_embeddings()
    print(f"Gerando embeddings para {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"Índice FAISS salvo em: {FAISS_INDEX_DIR}")


def main():
    print("1/3 — Carregando e limpando textos brutos...")
    documents = load_and_clean_documents()
    print(f"   {len(documents)} documentos válidos após limpeza.")

    print("2/3 — Quebrando em chunks...")
    chunks = chunk_documents(documents)
    print(f"   {len(chunks)} chunks gerados (tamanho alvo: {CHUNK_SIZE} caracteres, overlap: {CHUNK_OVERLAP}).")

    print("3/3 — Gerando embeddings e montando índice vetorial FAISS...")
    build_and_save_index(chunks)

    print("\nConcluído! Índice pronto para ser usado no grafo de RAG.")


if __name__ == "__main__":
    main()