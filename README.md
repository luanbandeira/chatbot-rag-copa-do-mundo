# Chatbot RAG — Copa do Mundo FIFA

Chatbot com Retrieval-Augmented Generation (RAG) sobre a história da Copa do
Mundo FIFA, desenvolvido para a atividade "Chatbot Simples com RAG".

## Base de conhecimento

- **Domínio:** Copa do Mundo FIFA (edições, seleções, recordes, história).
- **Fonte:** Wikipédia em português (pt.wikipedia.org), via API oficial de busca e extração de texto.
- **Volume:** 26 artigos completos (história geral, recordes, todas as edições de 1930 a 2026), somando ~1.346 chunks indexados.
- **Coleta:** `backend/app/scraper.py`
- **Textos brutos:** `backend/data/raw/`
- **Textos limpos (pós-processamento):** `backend/data/processed/`

## Arquitetura

- **Backend:** Python + FastAPI
- **Orquestração RAG:** LangGraph
- **Vector store:** FAISS (persistido em disco)
- **Embeddings:** modelo local multilíngue (`sentence-transformers`)
- **LLM de geração:** Groq (`openai/gpt-oss-20b`)
- **Frontend:** Next.js

## Como executar

### Backend

1. Entre na pasta `backend` e crie o ambiente virtual:
```
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows (PowerShell)
```
2. Instale o torch (CPU) antes do resto, para evitar baixar pacotes CUDA desnecessários:
```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
```
3. Copie `.env.example` para `.env` e preencha `GROQ_API_KEY` com sua chave (console.groq.com/keys).
4. Colete o conteúdo da base de conhecimento (Wikipédia PT sobre a Copa do Mundo):
```
   python -m app.scraper
```
5. Gere os embeddings e monte o índice vetorial FAISS:
```
   python -m app.ingest
```
6. Suba a API:
```
   uvicorn app.main:app --reload --port 8000
```
7. Teste o endpoint de chat:
```
   POST http://localhost:8000/chat
   Body: {"question": "Quem venceu a Copa do Mundo de 1970?", "history": []}
```

### Frontend

_(em construção)_

## Dependências

Ver `backend/requirements.txt` e `frontend/package.json`.

## Equipe

- Luan Bandeira de Melo Ramos
- Alexis Freitas
- Levi Arruda