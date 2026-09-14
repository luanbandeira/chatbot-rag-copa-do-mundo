"""
Configuração central da aplicação.
Carrega variáveis de ambiente do arquivo .env e expõe como constantes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"

if not GROQ_API_KEY:
    print(
        "[AVISO] GROQ_API_KEY não configurada. "
        "Copie .env.example para .env e preencha sua chave."
    )
