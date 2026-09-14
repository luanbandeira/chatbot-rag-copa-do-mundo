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

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


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

    text = "\n".join(lines)
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
    return splitter.split_documents(documents)


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