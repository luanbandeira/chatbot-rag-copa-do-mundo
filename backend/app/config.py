"""
Configuração central da aplicação.
Carrega variáveis de ambiente do arquivo .env e expõe como constantes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")

RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"

if not GROQ_API_KEY:
    print(
        "[AVISO] GROQ_API_KEY não configurada. "
        "Copie .env.example para .env e preencha sua chave."
    )