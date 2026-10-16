"use client";

import { useState } from "react";

type ChatMessage = { id: string; role: "user" | "assistant"; content: string };
const conversations = [
  { id: "today-1", title: "Plan my week", group: "Today" },
  { id: "today-2", title: "Understanding neural networks", group: "Today" },
  { id: "yesterday-1", title: "A better morning routine", group: "Yesterday" },
  { id: "yesterday-2", title: "Ideas for a small project", group: "Yesterday" },
  { id: "older-1", title: "Learning Python step by step", group: "Previous 7 days" },
];
const starters = [
  { icon: "✳", title: "Explore an idea", detail: "Help me think something through", prompt: "Help me explore an idea and think it through." },
  { icon: "⌘", title: "Build something", detail: "Turn a rough idea into a plan", prompt: "Help me turn a rough idea into a practical plan." },
  { icon: "↗", title: "Learn a topic", detail: "Explain it clearly, step by step", prompt: "Teach me a topic clearly, one step at a time." },
  { icon: "◷", title: "Get organized", detail: "Make a realistic plan for my day", prompt: "Help me make a realistic plan for my day." },
];

export function ChatShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeConversation, setActiveConversation] = useState("");
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [streamError, setStreamError] = useState("");

  function startConversation(title = "New conversation") {
    setActiveConversation(title);
    setMessages([]);
    setDraft("");
    setStreamError("");
    setSidebarOpen(false);
  }

  async function sendMessage(text: string) {
    const content = text.trim();
    if (!content || busy) return;
    const userId = crypto.randomUUID();
    const assistantId = crypto.randomUUID();
    setActiveConversation((current) => current || content.slice(0, 42));
    setMessages((current) => [
      ...current,
      { id: userId, role: "user", content },
      { id: assistantId, role: "assistant", content: "" },
    ]);
    setDraft("");
    setBusy(true);
    setStreamError("");

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ message: content }),
      });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => null) as { error?: string } | null;
        throw new Error(payload?.error || "Could not start the inference stream");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;
      while (!finished) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.split("\n").find((part) => part.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6)) as { type: string; token?: string; error?: string };
          if (event.type === "token" && event.token) {
            setMessages((current) => current.map((item) =>
              item.id === assistantId ? { ...item, content: item.content + event.token } : item,
            ));
          }
          if (event.type === "error") throw new Error(event.error || "Inference service failed");
          if (event.type === "done") finished = true;
        }
        if (done) break;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Stream failed";
      setStreamError(message);
      setMessages((current) => current.map((item) =>
        item.id === assistantId && !item.content ? { ...item, content: "The local preview could not respond. Please try again." } : item,
      ));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="workspace">
      {sidebarOpen && <button className="backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <aside className={"sidebar" + (sidebarOpen ? " open" : "")} aria-label="Chat navigation">
        <div className="brand-row">
          <a className="brand" href="/"><span className="brand-mark">t</span>TARA</a>
          <button className="icon mobile-close" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}>×</button>
          <button className="icon compose" aria-label="New chat" onClick={() => startConversation()}>↗</button>
        </div>
        <button className="new-chat" onClick={() => startConversation()}>＋ <span>New chat</span><kbd>⌘ K</kbd></button>
        <nav className="history" aria-label="Recent conversations">
          {["Today", "Yesterday", "Previous 7 days"].map((group) => (
            <section key={group}>
              <h2>{group}</h2>
              {conversations.filter((item) => item.group === group).map((item) => (
                <button key={item.id} className={"history-item" + (activeConversation === item.title ? " selected" : "")} onClick={() => startConversation(item.title)}>
                  <span className="dot" /> <span>{item.title}</span>
                </button>
              ))}
            </section>
          ))}
        </nav>
        <div className="account"><span className="avatar">L</span><span className="account-copy"><b>Local workspace</b><small>Personal account</small></span><button className="icon" aria-label="Account options">···</button></div>
      </aside>

      <section className="chat">
        <header className="topbar">
          <div className="top-left"><button className="icon mobile-menu" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}>☰</button><b>TARA <span className="muted">/</span> <span className="muted">{activeConversation || "Workspace"}</span></b></div>
          <span className="status"><i /> Provider-backed stream</span>
        </header>

        <div className="chat-scroll">
          {messages.length === 0 ? (
            <section className="welcome">
              <div className="orb" aria-hidden="true"><span>t</span><i>✳</i></div>
              <p className="eyebrow">YOUR THINKING SPACE</p>
              <h1>{activeConversation || <>Good to have you here<span className="lime">.</span></>}</h1>
              <p className="intro">A little curiosity can take you a long way. What would you like to work through today?</p>
              <div className="starter-grid">
                {starters.map((starter) => (
                  <button className="starter" key={starter.title} onClick={() => { setActiveConversation(starter.title); setDraft(starter.prompt); }}>
                    <span className="starter-icon">{starter.icon}</span><b>{starter.title}</b><small>{starter.detail}</small><span className="arrow">↗</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <section className="transcript" aria-label="Conversation" aria-live="polite">
              {messages.map((message) => (
                <article className={"message message-" + message.role} key={message.id}>
                  <span className="message-avatar">{message.role === "user" ? "Y" : "t"}</span>
                  <div className="message-body"><b>{message.role === "user" ? "You" : "TARA · model"}</b><p>{message.content || (busy && message.role === "assistant" ? <span className="typing">Thinking<span>.</span><span>.</span><span>.</span></span> : "")}</p></div>
                </article>
              ))}
              {streamError && <p className="stream-error" role="status">{streamError}</p>}
            </section>
          )}
        </div>

        <div className="composer-wrap">
          <form className="composer" onSubmit={(event) => { event.preventDefault(); void sendMessage(draft); }}>
            <textarea aria-label="Message TARA" placeholder="Message TARA..." rows={2} value={draft} disabled={busy} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(draft); } }} />
            <div className="composer-bottom"><button type="button" className="attach" aria-label="Add attachment" title="Attachments arrive in a later milestone">＋</button><span className="hint">{busy ? "Streaming model response…" : "Shift + Enter for a new line"}</span><button className="send" type="submit" aria-label="Send message" disabled={!draft.trim() || busy}>{busy ? "…" : "↑"}</button></div>
          </form>
          <p className="disclaimer">Uses the configured Groq model. Conversations are not saved by this chat interface.</p>
        </div>
      </section>
    </main>
  );
}
