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

### Pré-requisitos

- **Python 3.12.** Python 3.13 **não funciona**: algumas dependências (`numpy` 1.26 e `faiss-cpu` 1.9) não têm pacote para ele e a instalação falha. Confira com `py -3.12 --version` (Windows) ou `python3.12 --version` (Linux/macOS).
- **Node.js 20 ou mais recente** (confira com `node -v`).
- **Chave da Groq**, gratuita: crie uma conta em https://console.groq.com e gere a chave em https://console.groq.com/keys.
- ~2GB livres em disco (dependências + modelo de embeddings) e internet na primeira execução.

A aplicação usa **dois terminais**: um para o backend e outro para o frontend. Os comandos abaixo partem da pasta raiz do projeto.

### Backend (terminal 1)

1. Entre na pasta `backend` e crie o ambiente virtual com o Python 3.12:
```
   cd backend
   py -3.12 -m venv venv          # Windows
   python3.12 -m venv venv        # Linux/macOS
```
2. Ative o ambiente virtual (o terminal passa a mostrar `(venv)` no início da linha):
```
   .\venv\Scripts\Activate.ps1    # Windows (PowerShell)
   venv\Scripts\activate.bat      # Windows (Prompt de Comando)
   source venv/bin/activate       # Linux/macOS
```
   Se o PowerShell mostrar o erro "a execução de scripts foi desabilitada neste sistema", libere só para o terminal atual e ative de novo:
```
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
```
3. Instale o torch (versão CPU) antes do resto, para evitar baixar pacotes CUDA desnecessários, e depois as demais dependências:
```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
```
4. Crie o arquivo `.env` a partir do exemplo:
```
   copy .env.example .env         # Windows
   cp .env.example .env           # Linux/macOS
```
   Abra o `.env` num editor de texto e cole sua chave em `GROQ_API_KEY=` (ex.: `GROQ_API_KEY=gsk_...`).
5. Gere os embeddings e monte o índice vetorial FAISS. É obrigatório na primeira execução (o índice não é versionado) e leva alguns minutos, pois baixa o modelo de embeddings (~470MB):
```
   python -m app.ingest
```
   Ao final deve aparecer `Concluído! Índice pronto para ser usado no grafo de RAG.`
6. Suba a API (ainda dentro de `backend`, com o ambiente virtual ativado):
```
   uvicorn app.main:app --port 8000
```
   Deixe esse terminal aberto. A API está pronta quando aparecer `Application startup complete`.
7. (Opcional) Teste a API pelo navegador em http://localhost:8000/docs: abra `POST /chat`, clique em **Try it out**, use o corpo abaixo e clique em **Execute**:
```
   {"question": "Quem venceu a Copa do Mundo de 2022?", "history": []}
```

(Opcional) Para coletar de novo o conteúdo da Wikipédia, rode `python -m app.scraper` antes do passo 5. Os textos já estão em `data/raw/`, então isso não é necessário.

### Frontend (terminal 2)

Com o backend rodando, abra outro terminal na pasta raiz do projeto:

1. Entre na pasta `frontend` e instale as dependências:
```
   cd frontend
   npm install
```
2. Suba a interface:
```
   npm run dev
```
3. Acesse no navegador o endereço mostrado no terminal (normalmente http://localhost:3000) e faça perguntas, por exemplo: "Quem venceu a Copa de 2022?" e depois "E em 2018?".

Por padrão o frontend procura o backend em `http://localhost:8000`. Se o backend estiver em outro endereço, copie `.env.local.example` para `.env.local`, ajuste `NEXT_PUBLIC_API_URL` e reinicie o `npm run dev`.

### Nas próximas execuções

Não é preciso instalar nada nem rodar a ingestão de novo. Basta:

- terminal 1: `cd backend`, ativar o ambiente virtual (passo 2) e `uvicorn app.main:app --port 8000`;
- terminal 2: `cd frontend` e `npm run dev`.

## Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `pip install -r requirements.txt` falha tentando compilar `numpy` ou `faiss-cpu` | O ambiente virtual foi criado com Python diferente do 3.12. Apague a pasta `venv` e refaça o passo 1 com `py -3.12` / `python3.12`. |
| `[AVISO] GROQ_API_KEY não configurada` | O `.env` não existe ou está sem a chave, ou o comando foi rodado fora da pasta `backend` (o `.env` é lido da pasta atual). |
| `could not open data\faiss_index\index.faiss` | O índice ainda não foi gerado: rode `python -m app.ingest` (passo 5). |
| No chat aparece "Não consegui falar com o servidor do chatbot" | O backend não está rodando ou não está na porta 8000. Confira o terminal 1 e http://localhost:8000/health. |
| Aviso do Hugging Face sobre *symlinks* no Windows | Pode ser ignorado; afeta só o jeito como o modelo fica em cache. |
| A primeira pergunta demora alguns segundos a mais | Normal: o modelo de embeddings e o índice são carregados na primeira requisição. |
| `address already in use` na porta 8000, ou a API responde com código antigo | Um uvicorn antigo continua rodando. No Windows, veja o PID com `netstat -ano \| findstr :8000` e encerre com `taskkill /PID <número> /F`; no Linux/macOS, use `lsof -i :8000` e `kill <número>`. |

## Dependências

- Backend: `backend/requirements.txt` (+ `torch` CPU, instalado à parte conforme o passo 3)
- Frontend: `frontend/package.json`

## Equipe

- Luan Bandeira de Melo Ramos
- Alexis Freitas
- Levi Arruda
