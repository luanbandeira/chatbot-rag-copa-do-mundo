# Chatbot RAG — Copa do Mundo FIFA

Chatbot com Retrieval-Augmented Generation (RAG) sobre a história da Copa do
Mundo FIFA, desenvolvido para a atividade "Chatbot Simples com RAG".

## Base de conhecimento

_(em construção — será preenchido na etapa de coleta de conteúdo)_

Domínio: Copa do Mundo FIFA (edições, seleções, recordes, regulamento).
Fontes: Wikipédia (PT), FIFA, CBF.

## Arquitetura

- **Backend:** Python + FastAPI
- **Orquestração RAG:** LangGraph
- **Vector store:** Chroma (persistido em disco)
- **Embeddings:** modelo local multilíngue (`sentence-transformers`)
- **LLM de geração:** Groq (Llama 3.x)
- **Frontend:** Next.js

## Como executar

_(em construção — será preenchido conforme cada etapa for finalizada)_

## Dependências

Ver `backend/requirements.txt` e `frontend/package.json`.

## Equipe

- Luan Bandeira de Melo Ramos
- Alexis Freitas
- Levi Arruda
