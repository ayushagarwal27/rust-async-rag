import { useState, useEffect, useRef } from "react";

// Persist session ID in localStorage so history survives page reloads.
function getOrCreateSessionId() {
  const key = "rag_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        <p className="bubble-text">{msg.content}</p>
        {msg.sources && msg.sources.length > 0 && (
          <details className="sources">
            <summary>Sources ({msg.sources.length})</summary>
            <ul>
              {msg.sources.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [sessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const bottomRef = useRef(null);

  // Load conversation history on mount
  useEffect(() => {
    fetch(`/api/chat/history/${sessionId}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.messages?.length > 0) {
          setMessages(data.messages.map((m) => ({ ...m, sources: [] })));
        }
      })
      .catch(() => {})
      .finally(() => setLoadingHistory(false));
  }, [sessionId]);

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function newSession() {
    const key = "rag_session_id";
    const id = crypto.randomUUID();
    localStorage.setItem(key, id);
    // Reload so the sessionId state re-initialises
    window.location.reload();
  }

  async function sendMessage(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming) return;

    setInput("");
    setStreaming(true);

    // Append user message immediately
    setMessages((prev) => [...prev, { role: "user", content: text }]);

    // Placeholder for the streaming assistant reply
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", sources: [] },
    ]);

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.token) {
            // Append token to the last (assistant) message
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                content: last.content + data.token,
              };
              return updated;
            });
          } else if (data.done) {
            // Attach sources to the last message
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                sources: data.sources ?? [],
              };
              return updated;
            });
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message}`,
          sources: [],
        };
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-title">
          <span className="header-icon">🦀</span>
          <h1>Async Rust RAG</h1>
        </div>
        <div className="header-meta">
          <span className="session-label" title={sessionId}>
            Session: {sessionId.slice(0, 8)}…
          </span>
          <button className="new-session-btn" onClick={newSession}>
            New session
          </button>
        </div>
      </header>

      <main className="messages">
        {loadingHistory ? (
          <div className="empty-state">
            <p className="example-hint">Loading history…</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="empty-state">
            <p>Ask anything about async Rust — tokio, tracing, futures.</p>
            <p className="example-hint">
              Try: <em>"What is the difference between spawn and spawn_blocking?"</em>
            </p>
          </div>
        ) : null}
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {streaming && (
          <div className="message-row assistant-row">
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form className="input-bar" onSubmit={sendMessage}>
        <input
          className="input-field"
          type="text"
          placeholder="Ask about async Rust…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming || loadingHistory}
        />
        <button className="send-btn" type="submit" disabled={streaming || loadingHistory || !input.trim()}>
          {streaming ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
