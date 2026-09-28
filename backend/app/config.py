"""
Configuração central da aplicação.
Carrega variáveis de ambiente do arquivo .env e expõe como constantes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")

RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"

# Parte 2 — Engenharia de Prompt: limiar de distância L2 do FAISS abaixo do
# qual um chunk é considerado evidência relevante. Distância L2 cresce com a
# dissimilaridade (0 = idêntico); valores foram calibrados observando chunks
# de assunto correto (~0.6-1.1) vs. chunks de assunto não relacionado (>1.3)
# no nosso índice com o modelo paraphrase-multilingual-MiniLM-L12-v2.
EVIDENCE_DISTANCE_THRESHOLD = float(os.getenv("EVIDENCE_DISTANCE_THRESHOLD", "1.3"))

# Liga a variante few-shot do prompt de geração (ver app/prompts.py). Fica
# como variável de ambiente para o teste comparativo zero-shot vs few-shot
# não exigir editar código — só trocar no .env e reiniciar o servidor.
GENERATE_FEW_SHOT = os.getenv("GENERATE_FEW_SHOT", "false").strip().lower() == "true"

if not GROQ_API_KEY:
    print(
        "[AVISO] GROQ_API_KEY não configurada. "
        "Copie .env.example para .env e preencha sua chave."
    )
