import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Activity, ArrowDown, ArrowUp, Check, ChevronDown, CircleHelp, Code2, Command, FileText, Globe2, Menu, MessageSquare, Moon, Paperclip, Plus, RefreshCw, Settings2, ShieldCheck, Sparkles, Square, Sun, X, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Message = { id: string; role: "user" | "assistant"; content: string };
type RequestState = "idle" | "pending" | "error";
const API_BASE = (import.meta.env.VITE_TARA_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

const starterPrompts = [
  { icon: Code2, title: "Help with code", text: "Debug or write code", prompt: "Help me write and debug code." },
  { icon: Command, title: "Plan a project", text: "Break it into steps", prompt: "Help me break a project into clear steps." },
  { icon: Globe2, title: "Explore an idea", text: "Reason through a question", prompt: "Help me reason through a complex idea." },
];

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [lightMode, setLightMode] = useState(false);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [copiedId, setCopiedId] = useState("");
  const [lastSubmitted, setLastSubmitted] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const isBusy = requestState === "pending";

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/api/health`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error("Health check failed"); return response.json(); })
      .then(() => setBackendStatus("online"))
      .catch(() => { if (!controller.signal.aborted) setBackendStatus("offline"); });
    return () => controller.abort();
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, requestState]);

  function newChat() {
    abortRef.current?.abort();
    setMessages([]);
    setLastSubmitted("");
    setDraft("");
    setRequestState("idle");
    setError("");
    setSidebarOpen(false);
    inputRef.current?.focus();
  }

  async function sendMessage(raw = draft) {
    const text = raw.trim();
    if (!text || isBusy) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((current) => [...current, userMessage]);
    setLastSubmitted(text);
    setDraft("");
    setError("");
    setRequestState("pending");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      });
      const data: unknown = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof data === "object" && data !== null && "detail" in data && typeof data.detail === "string" ? data.detail : "";
        throw new Error(detail || `TARA backend returned HTTP ${response.status}.`);
      }
      if (typeof data !== "object" || data === null || !("message" in data) || typeof data.message !== "string") {
        throw new Error("The backend response did not contain a valid message.");
      }
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: data.message as string }]);
      setRequestState("idle");
      setBackendStatus("online");
    } catch (err) {
      if (controller.signal.aborted) {
        setRequestState("idle");
        return;
      }
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
      setRequestState("error");
      setBackendStatus("offline");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function stopRequest() {
    abortRef.current?.abort();
    abortRef.current = null;
    setRequestState("idle");
  }

  function submit(event: FormEvent) { event.preventDefault(); void sendMessage(); }
  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function copyMessage(message: Message) {
    try { await navigator.clipboard.writeText(message.content); setCopiedId(message.id); window.setTimeout(() => setCopiedId(""), 1400); } catch { /* clipboard may be unavailable */ }
  }

  return (
    <div className={lightMode ? "app light" : "app"}>
      <aside className={sidebarOpen ? "sidebar open" : "sidebar"}>
        <div className="brand"><div className="brand-mark"><Sparkles size={19} /></div><div><strong>TARA</strong><span>AI WORKSPACE</span></div><button className="icon-btn mobile-close" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><X size={18} /></button></div>
        <button className="new-chat" onClick={newChat}><Plus size={17} /> <span>New chat</span><kbd>⌘ K</kbd></button>
        <div className="side-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          <button className="nav-item active" onClick={() => setSidebarOpen(false)}><MessageSquare size={17} /><span>Chat</span></button>
          <button className="nav-item" onClick={() => setError("Conversation history will be available when a persistent history store is implemented.")}><FileText size={17} /><span>Recent chats</span><span className="nav-soon">Soon</span></button>
          <button className="nav-item" onClick={() => setError("Reasoning details are not yet exposed by the current chat API.")}><Activity size={17} /><span>Reasoning</span></button>
        </nav>
        <div className="side-label side-label-spaced">TOOLS</div>
        <nav className="nav-list" aria-label="Tools">
          <button className="nav-item" onClick={() => void sendMessage("Help me with Python.")}><Code2 size={17} /><span>Python help</span></button>
          <button className="nav-item" onClick={() => void sendMessage("Help me plan a project.")}><Zap size={17} /><span>Project planning</span></button>
        </nav>
        <div className="sidebar-bottom">
          <div className="backend-indicator"><span className={`status-dot ${backendStatus}`} /><div><strong>{backendStatus === "checking" ? "Checking backend" : backendStatus === "online" ? "Backend reachable" : "Backend unavailable"}</strong><span>{backendStatus === "online" ? "Health check passed" : backendStatus === "checking" ? "Verifying connection…" : "Start API to chat"}</span></div><button className="icon-btn" aria-label="Recheck backend" onClick={() => { setBackendStatus("checking"); fetch(`${API_BASE}/api/health`).then(r => { if (!r.ok) throw new Error(); setBackendStatus("online"); }).catch(() => setBackendStatus("offline")); }}><RefreshCw size={14} /></button></div>
          <div className="side-footer"><button className="nav-item" onClick={() => setError("Settings will be added after backend configuration is finalized.")}><Settings2 size={17} /><span>Settings</span></button><button className="nav-item" onClick={() => setError("TARA Chat MVP · API and persistent history depend on backend implementation.")}><CircleHelp size={17} /><span>Help & status</span></button></div>
          <div className="profile"><div className="avatar">T</div><div><strong>Local workspace</strong><span>Private session</span></div><ShieldCheck size={16} className="profile-shield" /></div>
        </div>
      </aside>
      {sidebarOpen && <button className="scrim" aria-label="Close navigation overlay" onClick={() => setSidebarOpen(false)} />}
      <main className="main">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-btn mobile-menu" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><span className="crumb-sep">/</span><strong>{messages.length ? "Conversation" : "New chat"}</strong></div></div>
          <div className="topbar-right"><div className="model-pill"><span className={`status-dot ${backendStatus}`} />{backendStatus === "online" ? "Backend connected" : backendStatus === "checking" ? "Checking status" : "Not connected"}<ChevronDown size={14} /></div><button className="icon-btn" aria-label={lightMode ? "Switch to dark theme" : "Switch to light theme"} onClick={() => setLightMode(v => !v)}>{lightMode ? <Moon size={17} /> : <Sun size={17} />}</button></div>
        </header>
        <section className={messages.length ? "conversation has-messages" : "conversation empty"}>
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-orb"><div className="orb-inner"><Sparkles size={29} /></div></div>
              <div className="eyebrow"><span className="eyebrow-line" /> YOUR REASONING WORKSPACE <span className="eyebrow-line" /></div>
              <h1>What shall we <span>figure out?</span></h1>
              <p className="welcome-copy">Think clearly. Explore ideas. Build something meaningful.</p>
              <div className="starter-grid">{starterPrompts.map(({ icon: Icon, title, text: sub, prompt }) => <button className="starter-card" key={title} onClick={() => { setDraft(prompt); inputRef.current?.focus(); }}><span className="starter-icon"><Icon size={18} /></span><strong>{title}</strong><span>{sub}</span><ArrowDown size={14} className="starter-arrow" /></button>)}</div>
            </div>
          ) : (
            <div className="message-list">{messages.map(message => <article className={`message ${message.role}`} key={message.id}><div className={message.role === "assistant" ? "message-avatar tara-avatar" : "message-avatar user-avatar"}>{message.role === "assistant" ? <Sparkles size={16} /> : "Y"}</div><div className="message-body"><div className="message-meta"><strong>{message.role === "assistant" ? "TARA" : "You"}</strong><span>{message.role === "assistant" ? "Assistant" : "You"}</span>{message.role === "assistant" && <button className="copy-btn" onClick={() => void copyMessage(message)} aria-label="Copy assistant message">{copiedId === message.id ? <Check size={14} /> : <FileText size={14} />}{copiedId === message.id ? "Copied" : "Copy"}</button>}</div><div className="message-content markdown-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div></div></article>)}
              {isBusy && <article className="message assistant"><div className="message-avatar tara-avatar"><Sparkles size={16} /></div><div className="message-body"><div className="message-meta"><strong>TARA</strong><span>Working on your message</span></div><div className="typing"><span /><span /><span /> <span className="typing-label">Waiting for response…</span></div></div></article>}
              <div ref={endRef} />
            </div>
          )}
          {error && <div className="error-banner" role="alert"><div className="error-icon"><Activity size={17} /></div><div className="error-text"><strong>We couldn't complete that request</strong><span>{error}</span><button className="retry-btn" onClick={() => { setError(""); void sendMessage(lastSubmitted); }}> <RefreshCw size={13} /> Try again</button></div><button className="icon-btn" aria-label="Dismiss error" onClick={() => setError("")}><X size={16} /></button></div>}
        </section>
        <footer className="composer-area">
          <form className="composer" onSubmit={submit}><textarea ref={inputRef} aria-label="Message TARA" value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={onComposerKeyDown} placeholder="Message TARA…  (Shift + Enter for new line)" rows={1} disabled={isBusy} onInput={e => { const el = e.currentTarget; el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 180)}px`; }} />
            <div className="composer-bottom"><div className="composer-tools"><button type="button" className="icon-btn" aria-label="Attachments are not available yet" title="Attachments coming later" onClick={() => setError("File attachments are not part of the initial MVP.")}><Paperclip size={17} /></button><span className="composer-hint"><ShieldCheck size={13} /> Responses come from your configured TARA backend</span></div><div className="composer-actions">{isBusy && <button type="button" className="stop-btn" onClick={stopRequest}><Square size={13} fill="currentColor" /> Stop</button>}<button type="submit" className="send-btn" disabled={!draft.trim() || isBusy} aria-label="Send message">{isBusy ? <span className="send-loading" /> : <ArrowUp size={19} />}</button></div></div>
          </form>
          <p className="disclaimer"><ShieldCheck size={12} /> TARA can make mistakes. Verify important information.</p>
        </footer>
      </main>
    </div>
  );
}

export default App;