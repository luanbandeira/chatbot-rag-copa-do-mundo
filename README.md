# Chatbot RAG — Copa do Mundo FIFA

Chatbot com Retrieval-Augmented Generation (RAG) sobre a história da Copa do
Mundo FIFA, desenvolvido para a atividade "Chatbot Simples com RAG".

## Base de conhecimento

- **Domínio:** Copa do Mundo FIFA (edições, seleções, recordes, história).
- **Fonte:** Wikipédia em português (pt.wikipedia.org), via API oficial de busca e extração de texto puro.
- **Volume:** 26 artigos completos, ~110 mil palavras (~220 páginas), somando ~1.228 chunks indexados:
  - artigo geral "Copa do Mundo FIFA" e "Recordes da Copa do Mundo FIFA";
  - todas as edições disputadas, de 1930 a 2022 (22 artigos);
  - as próximas edições, 2026 e 2030.
- **Coleta:** `backend/app/scraper.py`
- **Textos brutos:** `backend/data/raw/` (28 artigos coletados; 2 listas compostas quase só por tabelas são descartadas na limpeza)
- **Textos limpos (pós-processamento):** `backend/data/processed/`

Os textos coletados já estão no repositório, então não é preciso rodar o scraper de novo para executar o projeto.

## Arquitetura

- **Backend:** Python + FastAPI (`POST /chat`, `GET /health`)
- **Orquestração RAG:** LangGraph (`backend/app/graph.py`)
- **Vector store:** FAISS (persistido em disco)
- **Embeddings:** modelo local multilíngue `intfloat/multilingual-e5-small` (`sentence-transformers`)
- **LLM de geração:** Groq (`openai/gpt-oss-20b`), usada apenas na etapa de geração da resposta
- **Frontend:** Next.js (React, JavaScript)

### Pipeline

1. **Coleta** (`scraper.py`): baixa os artigos da Wikipédia.
2. **Limpeza e chunking** (`ingest.py`): remove seções de referências/links e cabeçalhos sem conteúdo, quebra em chunks de 800 caracteres (overlap de 120) e prefixa cada chunk com o título do artigo.
3. **Embeddings e índice** (`ingest.py`): gera os embeddings e salva o índice FAISS.
4. **Grafo LangGraph** (`graph.py`), executado a cada pergunta:
   1. `receive_question` — entrada da pergunta
   2. `retrieve_context` — recuperação dos 8 chunks mais relevantes (`retriever.py`)
   3. `build_prompt` — montagem do prompt com os chunks e as últimas mensagens da conversa
   4. `call_llm` — chamada da LLM (Groq)
   5. `format_response` — retorno da resposta com as fontes usadas
5. **Interface** (`frontend/`): envia a pergunta e o histórico para `/chat` e exibe a resposta.

## Como executar

Pré-requisitos: Python 3.12, Node.js 20+ e uma chave gratuita da Groq.

### Backend

1. Entre na pasta `backend` e crie o ambiente virtual:
```
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # Windows (PowerShell)
   source venv/bin/activate      # Linux/macOS
```
2. Instale o torch (CPU) antes do resto, para evitar baixar pacotes CUDA desnecessários:
```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
```
3. Copie `.env.example` para `.env` e preencha `GROQ_API_KEY` com sua chave (console.groq.com/keys).
4. (Opcional) Colete de novo o conteúdo da base de conhecimento — os textos já estão em `data/raw/`:
```
   python -m app.scraper
```
5. Gere os embeddings e monte o índice vetorial FAISS (obrigatório na primeira execução, o índice não é versionado). Na primeira vez o modelo de embeddings (~470MB) é baixado automaticamente:
```
   python -m app.ingest
```
6. Suba a API:
```
   uvicorn app.main:app --reload --port 8000
```
7. (Opcional) Teste o endpoint direto:
```
   POST http://localhost:8000/chat
   Body: {"question": "Quem venceu a Copa do Mundo de 1970?", "history": []}
```

### Frontend

Com o backend rodando, em outro terminal:

1. Entre na pasta `frontend` e instale as dependências:
```
   cd frontend
   npm install
```
2. Copie `.env.local.example` para `.env.local` (define `NEXT_PUBLIC_API_URL=http://localhost:8000`, o endereço do backend).
3. Suba a interface:
```
   npm run dev
```
4. Acesse http://localhost:3000 e faça perguntas, por exemplo: "Quem venceu a Copa de 2022?" e depois "E em 2018?".

## Dependências

- Backend: `backend/requirements.txt` (+ `torch` CPU, instalado à parte conforme o passo 2)
- Frontend: `frontend/package.json`

## Equipe

- Luan Bandeira de Melo Ramos
- Alexis Freitas
- Levi Arruda
