"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

  async function sendMessage(event) {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    const history = messages.map(({ role, content }) => ({ role, content }));
    const userMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });

      if (!response.ok) {
        throw new Error(`O servidor respondeu com status ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Não consegui falar com o servidor do chatbot. Confirme se o backend está rodando em " +
            API_URL +
            ".",
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
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
