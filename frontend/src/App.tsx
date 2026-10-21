import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { Activity, ArrowDown, ArrowUp, Check, Cpu, Menu, MessageCircle, Plus, Search, Settings2, ShieldCheck, Sparkles, Trash2, X, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Role = "user" | "assistant";
type Message = { id: string; role: Role; content: string; createdAt: number };
type Conversation = { id: string; title: string; messages: Message[]; updatedAt: number };
type Health = { status: string; provider?: string; model?: string; ollama?: string; detail?: string };
const KEY = "tara-react-conversations-v1";
const id = () => globalThis.crypto?.randomUUID?.() ?? ("tara-" + Date.now() + "-" + Math.random().toString(36).slice(2));
const freshChat = (): Conversation => ({ id: id(), title: "New conversation", messages: [], updatedAt: Date.now() });
function readChats(): Conversation[] {
  try { const data: unknown = JSON.parse(localStorage.getItem(KEY) ?? "[]"); return Array.isArray(data) ? data.filter((x: any) => x && typeof x.id === "string" && Array.isArray(x.messages)) : []; }
  catch { return []; }
}
function titleFor(text: string) { const value = text.replace(/\s+/g, " ").trim(); return value.length > 36 ? value.slice(0, 33) + "…" : value || "New conversation"; }

export default function App() {
  const [chats, setChats] = useState<Conversation[]>(readChats);
  const [activeId, setActiveId] = useState(() => readChats()[0]?.id ?? "");
  const [draft, setDraft] = useState("");
  const [health, setHealth] = useState<Health>({ status: "checking" });
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [toast, setToast] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const active = chats.find((chat) => chat.id === activeId);
  const messages = active?.messages ?? [];
  const visibleChats = useMemo(() => chats.filter((chat) => chat.title.toLowerCase().includes(search.toLowerCase().trim())), [chats, search]);
  const connected = health.status === "online" || health.status === "ok";
  const model = health.model || "llama3.2:3b";

  useEffect(() => { localStorage.setItem(KEY, JSON.stringify(chats)); }, [chats]);
  useEffect(() => {
    let stopped = false;
    const check = async () => {
      try { const response = await fetch("/api/health", { signal: AbortSignal.timeout(3500) }); const data = await response.json() as Health; if (!stopped) setHealth(data); }
      catch { if (!stopped) setHealth({ status: "offline", detail: "TARA API is not responding" }); }
    };
    void check(); const timer = window.setInterval(check, 15000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages.length, sending]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 2500); return () => window.clearTimeout(timer); }, [toast]);

  function updateChat(chatId: string, change: (chat: Conversation) => Conversation) {
    setChats((current) => current.map((chat) => chat.id === chatId ? change(chat) : chat));
  }
  function newChat() {
    const chat = freshChat(); setChats((current) => [chat, ...current]); setActiveId(chat.id); setDraft(""); setSidebarOpen(false);
    window.setTimeout(() => input.current?.focus(), 50);
  }
  function deleteChat(chatId: string) {
    const next = chats.filter((chat) => chat.id !== chatId); setChats(next);
    if (activeId === chatId) setActiveId(next[0]?.id ?? "");
    setToast("Conversation deleted");
  }
  async function send(event?: FormEvent) {
    event?.preventDefault(); const content = draft.trim(); if (!content || sending) return;
    let chat = chats.find((item) => item.id === activeId);
    if (!chat) { chat = freshChat(); setActiveId(chat.id); setChats((current) => [chat!, ...current]); }
    const chatId = chat.id;
    const history = chat.messages.slice(-16).map(({ role, content: text }) => ({ role, content: text }));
    const userMessage: Message = { id: id(), role: "user", content, createdAt: Date.now() };
    updateChat(chatId, (item) => ({ ...item, title: item.messages.length ? item.title : titleFor(content), updatedAt: Date.now(), messages: [...item.messages, userMessage] }));
    setDraft(""); setSending(true); setSidebarOpen(false);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: content, history }) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed (" + response.status + ")");
      const reply: Message = { id: id(), role: "assistant", content: String(data.response ?? "TARA returned an empty response."), createdAt: Date.now() };
      updateChat(chatId, (item) => ({ ...item, updatedAt: Date.now(), messages: [...item.messages, reply] }));
      setHealth((current) => ({ ...current, status: "online" }));
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown connection error";
      const reply: Message = { id: id(), role: "assistant", content: "**TARA could not reach its inference provider.**\n\n" + detail + "\n\nCheck that the backend is running and the configured Ollama model is installed.", createdAt: Date.now() };
      updateChat(chatId, (item) => ({ ...item, updatedAt: Date.now(), messages: [...item.messages, reply] }));
      setHealth((current) => ({ ...current, status: "degraded", detail }));
    } finally { setSending(false); window.setTimeout(() => input.current?.focus(), 50); }
  }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void send(); } }

  return <div className="app-shell">
    {sidebarOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside className={"sidebar " + (sidebarOpen ? "sidebar-open" : "")}>
      <div className="brand-row"><div className="brand-mark"><Sparkles size={19}/></div><div className="brand-copy"><span>TARA</span><small>REASONING WORKSPACE</small></div><button className="icon-btn mobile-close" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)}><X size={18}/></button></div>
      <button className="new-chat-btn" onClick={newChat}><Plus size={17}/> New conversation <kbd>Ctrl K</kbd></button>
      <div className="section-label section-gap">WORKSPACE</div>
      <div className="nav-item nav-active"><MessageCircle size={17}/><span>Chat</span><span className="nav-count">{String(chats.length).padStart(2, "0")}</span></div>
      <button className="nav-item" onClick={() => { setSettingsOpen(true); setSidebarOpen(false); }}><Settings2 size={17}/><span>Connection settings</span></button>
      <div className="history-heading"><span className="section-label">RECENT CONVERSATIONS</span><Search size={14}/></div>
      <div className="history-search"><Search size={14}/><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search chats" aria-label="Search conversations"/>{search && <button className="icon-btn tiny" onClick={() => setSearch("")} aria-label="Clear search"><X size={13}/></button>}</div>
      <div className="conversation-list">{visibleChats.length === 0 ? <p className="empty-history">{search ? "No matching conversations" : "Your chats will appear here."}</p> : visibleChats.map((chat) =>
        <div key={chat.id} className={"conversation-row " + (chat.id === activeId ? "conversation-selected" : "")}>
          <button className="conversation-select" onClick={() => { setActiveId(chat.id); setSidebarOpen(false); }}><MessageCircle size={15}/><span>{chat.title}</span></button>
          <button className="delete-chat icon-btn tiny" aria-label={"Delete " + chat.title} onClick={() => deleteChat(chat.id)}><Trash2 size={13}/></button>
        </div>)}</div>
      <div className="sidebar-bottom"><div className="runtime-card">
        <div className="runtime-status"><span className={"status-dot " + (connected ? "status-online" : "")}/><span>{connected ? "RUNTIME CONNECTED" : health.status === "checking" ? "CHECKING RUNTIME" : "RUNTIME DISCONNECTED"}</span><Activity size={14}/></div>
        <div className="runtime-model"><Cpu size={15}/><span>{model}</span></div>
        <p>{connected ? "Messages are handled by TARA's Python backend." : "Start the backend and local inference provider to chat."}</p>
        <button className="runtime-link" onClick={() => setSettingsOpen(true)}>Connection details <ArrowUp size={13}/></button>
      </div><div className="privacy-note"><ShieldCheck size={15}/><span>Local-first · No API keys in browser</span></div></div>
    </aside>

    <main className="main-panel">
      <header className="topbar"><div className="topbar-left"><button className="icon-btn mobile-menu" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={19}/></button><span className="breadcrumb">Workspace</span><span className="crumb-separator">/</span><span className="breadcrumb-current">Chat</span></div>
        <div className="topbar-right"><div className={"connection-pill " + (connected ? "pill-connected" : "")}><span className={"status-dot " + (connected ? "status-online" : "")}/>{connected ? "Connected" : "Local runtime"}</div><button className="icon-btn" aria-label="Open settings" onClick={() => setSettingsOpen(true)}><Settings2 size={18}/></button><button className="top-new-chat" onClick={newChat}><Plus size={15}/> New chat</button></div>
      </header>
      <section className={"chat-page " + (messages.length ? "chat-has-messages" : "")}>
        {messages.length === 0 ? <div className="welcome-screen">
          <div className="welcome-orb"><div className="orb-inner"><Sparkles size={27}/></div><span className="orb-ring ring-one"/><span className="orb-ring ring-two"/></div>
          <div className="eyebrow"><span/> YOUR LOCAL REASONING PARTNER <span/></div>
          <h1>Think clearly.<br/><em>Build intelligently.</em></h1>
          <p className="welcome-description">A thoughtful workspace for exploring ideas, solving problems, and making progress — powered by your TARA runtime.</p>
          <div className="capability-row"><span><Zap size={14}/> Reasoning</span><span><Activity size={14}/> Context-aware</span><span><ShieldCheck size={14}/> Local-first</span></div>
          <div className="prompt-grid">{[["Explain TARA", "Explain TARA's architecture in simple terms."], ["Plan a project", "Help me break a coding project into practical steps."], ["Explore an idea", "Help me explore an idea and identify assumptions."]].map(([label, prompt]) =>
            <button key={label} className="prompt-card" onClick={() => { setDraft(prompt); input.current?.focus(); }}><span>{label}</span><ArrowDown size={15}/><small>{prompt}</small></button>)}</div>
        </div> : <div className="messages-wrap">{messages.map((message) =>
          <article key={message.id} className={"message-row message-" + message.role}><div className={"message-avatar " + (message.role === "assistant" ? "tara-avatar" : "")}>{message.role === "assistant" ? <Sparkles size={16}/> : "Y"}</div>
            <div className="message-content"><div className="message-meta"><strong>{message.role === "assistant" ? "TARA" : "You"}</strong><span>{new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div><div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div></div>
          </article>)}{sending && <div className="message-row message-assistant"><div className="message-avatar tara-avatar"><Sparkles size={16}/></div><div className="typing-indicator"><span/><span/><span/><small>TARA is thinking</small></div></div>}<div ref={bottom}/></div>}
      </section>
      <div className="composer-area"><form className="composer" onSubmit={(event) => void send(event)}>
        <textarea ref={input} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={onKeyDown} placeholder="Message TARA…" aria-label="Message TARA" rows={1} maxLength={12000} disabled={sending}/>
        <div className="composer-footer"><div className="composer-context"><span className={"status-dot " + (connected ? "status-online" : "")}/><span>{connected ? "Connected to " + model : "Connect TARA runtime to get real responses"}</span></div><div className="composer-actions"><span className="keyboard-hint">Enter to send · Shift + Enter for new line</span><button className="send-btn" type="submit" disabled={!draft.trim() || sending} aria-label="Send message">{sending ? <span className="send-spinner"/> : <ArrowUp size={19}/>}</button></div></div>
      </form><p className="disclaimer">TARA can make mistakes. Verify important information before acting on it.</p></div>
    </main>

    {settingsOpen && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setSettingsOpen(false); }}><section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="modal-heading"><div><span className="eyebrow-small">RUNTIME STATUS</span><h2 id="settings-title">Connection details</h2></div><button className="icon-btn" aria-label="Close settings" onClick={() => setSettingsOpen(false)}><X size={18}/></button></div>
      <div className="settings-status"><span className={"status-dot " + (connected ? "status-online" : "")}/><div><strong>{connected ? "TARA API is responding" : "TARA API is not connected"}</strong><p>{health.detail || (connected ? "The frontend can reach the Python backend." : "Start the backend and ensure Ollama is running.")}</p></div></div>
      <div className="settings-field"><label>Backend API</label><div className="readonly-field"><code>http://127.0.0.1:8000</code><Check size={15}/></div></div>
      <div className="settings-field"><label>Inference provider</label><div className="readonly-field"><span>{health.provider || "Ollama (local)"}</span><span className="muted-label">local backend</span></div></div>
      <div className="settings-field"><label>Model</label><div className="readonly-field"><span>{model}</span><span className="muted-label">from backend config</span></div></div>
      <div className="modal-note"><ShieldCheck size={17}/><p>Credentials stay on the backend. This interface never asks for or stores provider API keys.</p></div>
      <div className="modal-actions"><button className="secondary-btn" onClick={() => { setHealth({ status: "checking" }); fetch("/api/health").then((r) => r.json()).then(setHealth).catch(() => setHealth({ status: "offline", detail: "TARA API is not responding" })); }}>Test connection</button><button className="primary-btn" onClick={() => setSettingsOpen(false)}>Done</button></div>
    </section></div>}
    {toast && <div className="toast"><Check size={15}/>{toast}</div>}
  </div>;
}
