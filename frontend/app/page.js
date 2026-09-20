"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// O pipeline de RAG (embeddings + LLM) leva alguns segundos; sem um limite a
// requisição pode ficar pendurada indefinidamente se o backend travar.
const REQUEST_TIMEOUT_MS = 45000;

class HttpError extends Error {
  constructor(status) {
    super("HTTP " + status);
    this.name = "HttpError";
    this.status = status;
  }
}

// Cada falha tem uma causa e uma ação diferente para quem está usando o chat,
// então vale distingui-las em vez de mostrar sempre a mesma mensagem.
function describeError(error) {
  if (error.name === "AbortError") {
    const segundos = Math.round(REQUEST_TIMEOUT_MS / 1000);
    return (
      "A resposta passou de " +
      segundos +
      " segundos e a requisição foi cancelada. O modelo pode estar sobrecarregado."
    );
  }

  if (error.name === "HttpError") {
    if (error.status === 429) {
      return "O limite de requisições da API do modelo foi atingido. Espere alguns instantes antes de tentar de novo.";
    }
    if (error.status >= 500) {
      return (
        "O backend respondeu com erro interno (HTTP " +
        error.status +
        "). Confira o terminal do servidor: a GROQ_API_KEY está configurada e o índice FAISS já foi gerado com python -m app.ingest?"
      );
    }
    return "O backend recusou a requisição (HTTP " + error.status + ").";
  }

  return (
    "Não consegui falar com o servidor do chatbot em " +
    API_URL +
    ". Confirme se o backend está rodando."
  );
}

// Negrito (**texto**) e itálico (*texto* ou _texto_), que a LLM usa nas respostas
const INLINE_MARKDOWN = /\*\*([^*\n]+)\*\*|(?<![\w*])\*([^*\s][^*\n]*)\*(?![\w*])|(?<!\w)_([^_\n]+)_(?!\w)/g;

// Converte o markdown simples em elementos React, sem injetar HTML
function renderInlineMarkdown(text) {
  const parts = [];
  let lastIndex = 0;

  for (const match of text.matchAll(INLINE_MARKDOWN)) {
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index));

    const [, bold, italicStar, italicUnderscore] = match;
    parts.push(
      bold ? (
        <strong key={match.index}>{bold}</strong>
      ) : (
        <em key={match.index}>{italicStar ?? italicUnderscore}</em>
      )
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function askQuestion(question, previousMessages) {
    // O histórico enviado ao backend ignora as bolhas de erro: elas são avisos
    // da interface, não parte da conversa.
    const history = previousMessages
      .filter((message) => !message.isError)
      .map(({ role, content }) => ({ role, content }));

    setMessages([...previousMessages, { role: "user", content: question }]);
    setLoading(true);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new HttpError(response.status);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources ?? [] },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: describeError(error),
          isError: true,
          // Guardados para o botão "Tentar novamente" repetir a pergunta a
          // partir do mesmo ponto da conversa.
          retryQuestion: question,
          retryBase: previousMessages,
        },
      ]);
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  }

  function sendMessage(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setInput("");
    askQuestion(question, messages);
  }

  return (
    <main className="page">
      <header className="header">
        <h1>Chatbot Copa do Mundo</h1>
        <p>Pergunte sobre edições, seleções e recordes da Copa do Mundo FIFA</p>
      </header>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <strong>Comece a conversa</strong>
            Pergunte, por exemplo: &ldquo;Quem venceu a Copa de 1994?&rdquo; ou
            &ldquo;Quantos títulos o Brasil já tem?&rdquo;
          </div>
        )}

        {messages.map((message, index) => (
          <div key={index} className={`bubble-row ${message.role}`}>
            <div className={`bubble ${message.role} ${message.isError ? "error" : ""}`}>
              {message.role === "assistant" ? renderInlineMarkdown(message.content) : message.content}

              {message.sources?.length > 0 && (
                <div className="sources">
                  <span className="sources-label">Fontes</span>
                  {message.sources.map((source) => (
                    <span className="source-chip" key={source}>
                      {source}
                    </span>
                  ))}
                </div>
              )}

              {message.isError && (
                <button
                  type="button"
                  className="retry-button"
                  onClick={() => askQuestion(message.retryQuestion, message.retryBase)}
                  disabled={loading}
                >
                  Tentar novamente
                </button>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="bubble-row assistant">
            <div className="typing">Buscando na base de conhecimento...</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="input-bar" onSubmit={sendMessage}>
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Escreva sua pergunta sobre a Copa do Mundo..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()} aria-label="Enviar">
          ➤
        </button>
      </form>
    </main>
  );
}
