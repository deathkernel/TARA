import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { Activity, ArrowDown, ArrowUp, Check, ChevronDown, CircleHelp, Code2, Command, FileText, Globe2, Menu, MessageSquare, Moon, Paperclip, Plus, RefreshCw, Settings2, ShieldCheck, Sparkles, Square, Sun, X, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Conversation, HISTORY_KEY, Message, THEME_KEY, parseHistory } from "./chatStorage";
import { assistantMessageFromPayload, apiErrorFromPayload, healthStatusFromPayload, isHealthCheckCurrent, isRequestCurrent } from "./chatApi";
import { isNewChatShortcut, shouldSendOnEnter } from "./keyboard";
import { focusTrapAction } from "./dialogFocus";

function readTheme(): boolean {
  try { return localStorage.getItem(THEME_KEY) === "light"; } catch { return false; }
}
function readHistory(): Conversation[] {
  try { return parseHistory(localStorage.getItem(HISTORY_KEY)); } catch { return []; }
}
type RequestState = "idle" | "pending" | "error";
const API_BASE = (import.meta.env.VITE_TARA_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";

const starterPrompts = [
  { icon: Code2, title: "Help with code", text: "Debug or write code", prompt: "Help me write and debug code." },
  { icon: Command, title: "Plan a project", text: "Break it into steps", prompt: "Help me break a project into clear steps." },
  { icon: Globe2, title: "Explore an idea", text: "Reason through a question", prompt: "Help me reason through a complex idea." },
];

function App() {
  const [recentChats, setRecentChats] = useState<Conversation[]>(() => readHistory());
  const [activeChatId, setActiveChatId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [lightMode, setLightMode] = useState(readTheme);
  const [backendStatus, setBackendStatus] = useState<"checking" | "ready" | "needs-config" | "offline">("checking");
  const [copiedId, setCopiedId] = useState("");
  const [copyStatus, setCopyStatus] = useState("");
  const [lastSubmitted, setLastSubmitted] = useState("");
  const [retryAvailable, setRetryAvailable] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const modalRef = useRef<HTMLElement>(null);
  const modalTriggerRef = useRef<HTMLElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const healthCheckIdRef = useRef(0);
  const copyTimeoutRef = useRef<number | null>(null);
  const isBusy = requestState === "pending";

  async function checkBackendStatus(signal?: AbortSignal) {
    const checkId = ++healthCheckIdRef.current;
    setBackendStatus("checking");
    try {
      const response = await fetch(`${API_BASE}/api/health`, signal ? { signal } : undefined);
      if (!response.ok) throw new Error("Health check failed");
      const data: unknown = await response.json();
      const status = healthStatusFromPayload(data);
      if (!status) throw new Error("Unexpected health response");
      // Ignore older checks if a newer manual recheck has already started.
      if (!isHealthCheckCurrent(checkId, healthCheckIdRef.current, signal)) return;
      setBackendStatus(status);
    } catch {
      if (isHealthCheckCurrent(checkId, healthCheckIdRef.current, signal)) setBackendStatus("offline");
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    void checkBackendStatus(controller.signal);
    return () => controller.abort();
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, requestState]);

  useEffect(() => () => {
    if (copyTimeoutRef.current !== null) window.clearTimeout(copyTimeoutRef.current);
  }, []);

  useEffect(() => {
    // Keep the composer compact after sending, clearing, or switching chats.
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }, [draft]);

  useEffect(() => {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(recentChats.slice(0, 30))); } catch { /* storage may be disabled or full */ }
  }, [recentChats]);

  useEffect(() => {
    try { localStorage.setItem(THEME_KEY, lightMode ? "light" : "dark"); } catch { /* storage may be disabled */ }
  }, [lightMode]);

  useEffect(() => {
    // Keep open TARA tabs in sync when a user changes local history or theme elsewhere.
    function syncBrowserPreferences(event: StorageEvent) {
      // A null key means another tab called localStorage.clear().
      if (event.key === null || event.key === HISTORY_KEY) setRecentChats(readHistory());
      if (event.key === null || event.key === THEME_KEY) setLightMode(readTheme());
    }
    window.addEventListener("storage", syncBrowserPreferences);
    return () => window.removeEventListener("storage", syncBrowserPreferences);
  }, []);

  useEffect(() => {
    if (!messages.length) return;
    const id = activeChatId || crypto.randomUUID();
    if (!activeChatId) setActiveChatId(id);
    const firstUserMessage = messages.find((message) => message.role === "user")?.content ?? "New conversation";
    const title = firstUserMessage.length > 44 ? `${firstUserMessage.slice(0, 41)}…` : firstUserMessage;
    const conversation: Conversation = { id, title, updatedAt: Date.now(), messages };
    setRecentChats((current) => [conversation, ...current.filter((item) => item.id !== id)].slice(0, 30));
  }, [messages, activeChatId]);

  useEffect(() => {
    if (!settingsOpen && !helpOpen) return;
    const previousFocus = modalTriggerRef.current;
    const modal = modalRef.current;
    const firstFocusable = modal?.querySelector<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])');
    firstFocusable?.focus();

    function keepFocusInside(event: globalThis.KeyboardEvent) {
      if (event.key !== "Tab" || !modal) return;
      const focusable = Array.from(modal.querySelectorAll<HTMLElement>('button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'));
      const activeIndex = focusable.indexOf(document.activeElement as HTMLElement);
      const action = focusTrapAction(activeIndex, focusable.length, event.shiftKey);
      if (action === null) return;
      event.preventDefault();
      if (action === "container") modal.focus();
      else if (action === "first") focusable[0]?.focus();
      else focusable[focusable.length - 1]?.focus();
    }
    window.addEventListener("keydown", keepFocusInside);
    return () => {
      window.removeEventListener("keydown", keepFocusInside);
      previousFocus?.focus();
    };
  }, [settingsOpen, helpOpen]);

  useEffect(() => {
    function onGlobalKeyDown(event: globalThis.KeyboardEvent) {
      if (isNewChatShortcut(event)) {
        event.preventDefault();
        newChat();
      }
      if (event.key === "Escape") {
        setSidebarOpen(false);
        setSettingsOpen(false);
        setHelpOpen(false);
      }
    }
    window.addEventListener("keydown", onGlobalKeyDown);
    return () => window.removeEventListener("keydown", onGlobalKeyDown);
  }, []);

  function newChat() {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setMessages([]);
    setActiveChatId("");
    setLastSubmitted("");
    setRetryAvailable(false);
    setDraft("");
    setRequestState("idle");
    setError("");
    setSidebarOpen(false);
    inputRef.current?.focus();
  }

  async function sendMessage(raw = draft, addUserMessage = true) {
    const text = raw.trim();
    // The ref blocks rapid duplicate submits before React has rendered the pending state.
    if (!text || isBusy || abortRef.current) return;
    if (addUserMessage) {
      const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: text };
      setMessages((current) => [...current, userMessage]);
      setLastSubmitted(text);
      setDraft("");
    }
    setError("");
    setRetryAvailable(false);
    setRequestState("pending");
    const controller = new AbortController();
    const requestId = ++requestIdRef.current;
    abortRef.current = controller;
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      });
      const data: unknown = await response.json().catch(() => ({}));
      if (!isRequestCurrent(requestId, requestIdRef.current, controller.signal)) return;
      if (!response.ok) throw new Error(apiErrorFromPayload(response.status, data));
      const assistantMessage = assistantMessageFromPayload(data);
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: assistantMessage }]);
      setRequestState("idle");
      setRetryAvailable(false);
      setBackendStatus("ready");
    } catch (err) {
      if (!isRequestCurrent(requestId, requestIdRef.current, controller.signal)) return;
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
      setRetryAvailable(true);
      setRequestState("error");
      if (err instanceof TypeError) setBackendStatus("offline");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function openConversation(conversation: Conversation) {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setActiveChatId(conversation.id);
    setMessages(conversation.messages);
    setLastSubmitted([...conversation.messages].reverse().find((message) => message.role === "user")?.content ?? "");
    setRequestState("idle");
    setError("");
    setRetryAvailable(false);
    setSidebarOpen(false);
  }

  function clearHistory() {
    // Clearing history also invalidates an in-flight response so it cannot
    // repopulate the conversation after the user has cleared their data.
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setRequestState("idle");
    setRecentChats([]);
    setActiveChatId("");
    setMessages([]);
    setLastSubmitted("");
    setRetryAvailable(false);
    setDraft("");
    setError("");
    setSettingsOpen(false);
    try { localStorage.removeItem(HISTORY_KEY); } catch { /* storage may be disabled */ }
  }

  function stopRequest() {
    requestIdRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    setRequestState("idle");
  }

  function submit(event: FormEvent) { event.preventDefault(); void sendMessage(); }
  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (shouldSendOnEnter(event.key, event.shiftKey, event.nativeEvent.isComposing)) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function copyMessage(message: Message) {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedId(message.id);
      setCopyStatus("Message copied to clipboard.");
    } catch {
      setCopiedId("");
      setCopyStatus("Clipboard access is unavailable. You can select and copy the message text manually.");
    }
    if (copyTimeoutRef.current !== null) window.clearTimeout(copyTimeoutRef.current);
    copyTimeoutRef.current = window.setTimeout(() => {
      setCopiedId("");
      setCopyStatus("");
      copyTimeoutRef.current = null;
    }, 3000);
  }

  return (
    <div className={lightMode ? "app light" : "app"}>
      <aside id="app-sidebar" className={sidebarOpen ? "sidebar open" : "sidebar"}>
        <div className="brand"><div className="brand-mark"><Sparkles size={19} /></div><div><strong>TARA</strong><span>AI WORKSPACE</span></div><button className="icon-btn mobile-close" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><X size={18} /></button></div>
        <button className="new-chat" onClick={newChat}><Plus size={17} /> <span>New chat</span><kbd>⌘ K</kbd></button>
        <div className="side-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          <button className="nav-item active" onClick={() => setSidebarOpen(false)}><MessageSquare size={17} /><span>Chat</span></button>
          <button className="nav-item" aria-expanded={historyExpanded} onClick={() => setHistoryExpanded((value) => !value)}><FileText size={17} /><span>Recent chats</span><span className="nav-soon">{recentChats.length}</span><ChevronDown size={13} className={historyExpanded ? "history-chevron expanded" : "history-chevron"} /></button>
          {historyExpanded && (recentChats.length ? recentChats.slice(0, 30).map((chat) => <button key={chat.id} className={`history-item${chat.id === activeChatId ? " selected" : ""}`} title={chat.title} onClick={() => openConversation(chat)}><MessageSquare size={13} /><span>{chat.title}</span></button>) : <p className="history-empty">Your conversations will appear here.</p>)}
          <button className="nav-item" onClick={() => { setError("Reasoning details are not yet exposed by the current chat API."); setRetryAvailable(false); }}><Activity size={17} /><span>Reasoning</span></button>
        </nav>
        <div className="side-label side-label-spaced">TOOLS</div>
        <nav className="nav-list" aria-label="Tools">
          <button className="nav-item" onClick={() => void sendMessage("Help me with Python.")}><Code2 size={17} /><span>Python help</span></button>
          <button className="nav-item" onClick={() => void sendMessage("Help me plan a project.")}><Zap size={17} /><span>Project planning</span></button>
        </nav>
        <div className="sidebar-bottom">
          <div className="backend-indicator" role="status" aria-live="polite" aria-atomic="true"><span className={`status-dot ${backendStatus}`} /><div><strong>{backendStatus === "checking" ? "Checking backend" : backendStatus === "ready" ? "Backend ready" : backendStatus === "needs-config" ? "Groq key needed" : "Backend unavailable"}</strong><span>{backendStatus === "ready" ? "Provider configured" : backendStatus === "needs-config" ? "Set GROQ_API_KEY on server" : backendStatus === "checking" ? "Verifying connection…" : "Start API to chat"}</span></div><button className="icon-btn" aria-label="Recheck backend" onClick={() => { void checkBackendStatus(); }}><RefreshCw size={14} /></button></div>
          <div className="side-footer"><button className="nav-item" onClick={(event) => { modalTriggerRef.current = event.currentTarget; setHelpOpen(false); setSettingsOpen(true); }}><Settings2 size={17} /><span>Settings</span></button><button className="nav-item" onClick={(event) => { modalTriggerRef.current = event.currentTarget; setSettingsOpen(false); setHelpOpen(true); }}><CircleHelp size={17} /><span>Help & status</span></button></div>
          <div className="profile"><div className="avatar">T</div><div><strong>Local workspace</strong><span>Private session</span></div><ShieldCheck size={16} className="profile-shield" /></div>
        </div>
      </aside>
      {sidebarOpen && <button className="scrim" aria-label="Close navigation overlay" onClick={() => setSidebarOpen(false)} />}
      {(settingsOpen || helpOpen) && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) { setSettingsOpen(false); setHelpOpen(false); } }}><section ref={modalRef} className="settings-modal" tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby="modal-title" aria-describedby="modal-description"><div className="modal-heading"><div><span className="modal-eyebrow">{settingsOpen ? "PREFERENCES" : "TARA WORKSPACE"}</span><h2 id="modal-title">{settingsOpen ? "Settings" : "Help & status"}</h2></div><button className="icon-btn" aria-label="Close dialog" onClick={() => { setSettingsOpen(false); setHelpOpen(false); }}><X size={18} /></button></div>{settingsOpen ? <><div className="setting-row"><div><strong>Appearance</strong><span>Choose the workspace color theme.</span></div><button className="setting-action" onClick={() => setLightMode((value) => !value)}>{lightMode ? "Switch to dark" : "Switch to light"}</button></div><div className="setting-row"><div><strong>Local chat history</strong><span>{recentChats.length} saved conversation{recentChats.length === 1 ? "" : "s"} on this browser.</span></div><button className="setting-action danger-action" onClick={clearHistory}>Clear history</button></div><p className="modal-note" id="modal-description">History is stored in this browser only. Clearing browser storage can remove it; it is not synced to an account.</p></> : <><div className="help-status"><span className={`status-dot ${backendStatus}`} /><div><strong>{backendStatus === "ready" ? "Backend reachable" : backendStatus === "needs-config" ? "Provider key missing" : backendStatus === "checking" ? "Checking backend" : "Backend unavailable"}</strong><span>{backendStatus === "ready" ? "The health check reports a configured provider." : backendStatus === "needs-config" ? "The API is reachable, but GROQ_API_KEY is not configured." : backendStatus === "checking" ? "Waiting for the health check." : "Check that the API server is running."}</span></div><button className="setting-action" onClick={() => { void checkBackendStatus(); }}>Recheck</button></div><p className="modal-note" id="modal-description">Keyboard: Enter sends, Shift+Enter adds a line, Ctrl/⌘+K starts a new chat, Escape closes dialogs and navigation.</p></>}</section></div>}
      <main className="main">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-btn mobile-menu" aria-label="Open navigation" aria-controls="app-sidebar" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(true)}><Menu size={19} /></button><div className="breadcrumb"><span>Workspace</span><span className="crumb-sep">/</span><strong>{messages.length ? "Conversation" : "New chat"}</strong></div></div>
          <div className="topbar-right"><div className="model-pill" role="status" aria-live="polite"><span className={`status-dot ${backendStatus}`} />{backendStatus === "ready" ? "Backend ready" : backendStatus === "needs-config" ? "Groq key needed" : backendStatus === "checking" ? "Checking status" : "Not connected"}<ChevronDown size={14} /></div><button className="icon-btn" aria-label={lightMode ? "Switch to dark theme" : "Switch to light theme"} onClick={() => setLightMode(v => !v)}>{lightMode ? <Moon size={17} /> : <Sun size={17} />}</button></div>
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
            <div className="message-list" aria-busy={isBusy}>{messages.map(message => <article className={`message ${message.role}`} key={message.id}><div className={message.role === "assistant" ? "message-avatar tara-avatar" : "message-avatar user-avatar"}>{message.role === "assistant" ? <Sparkles size={16} /> : "Y"}</div><div className="message-body"><div className="message-meta"><strong>{message.role === "assistant" ? "TARA" : "You"}</strong><span>{message.role === "assistant" ? "Assistant" : "You"}</span>{message.role === "assistant" && <button className="copy-btn" onClick={() => void copyMessage(message)} aria-label={copiedId === message.id ? "Message copied" : "Copy assistant message"}>{copiedId === message.id ? <Check size={14} /> : <FileText size={14} />}{copiedId === message.id ? "Copied" : "Copy"}</button>}</div><div className="message-content markdown-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div></div></article>)}
              {isBusy && <article className="message assistant"><div className="message-avatar tara-avatar"><Sparkles size={16} /></div><div className="message-body"><div className="message-meta"><strong>TARA</strong><span>Working on your message</span></div><div className="typing" role="status" aria-live="polite"><span /><span /><span /> <span className="typing-label">Waiting for response…</span></div></div></article>}
              <div ref={endRef} />
            </div>
          )}
          <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">{copyStatus}</p>
          {copyStatus && <p className={`copy-feedback ${copyStatus.startsWith("Clipboard access") ? "copy-feedback-error" : ""}`}>{copyStatus}</p>}
          {error && <div className="error-banner" role="alert"><div className="error-icon"><Activity size={17} /></div><div className="error-text"><strong>We couldn't complete that request</strong><span>{error}</span>{retryAvailable && lastSubmitted.trim() && <button className="retry-btn" onClick={() => { setError(""); void sendMessage(lastSubmitted, false); }}> <RefreshCw size={13} /> Try again</button>}</div><button className="icon-btn" aria-label="Dismiss error" onClick={() => setError("")}><X size={16} /></button></div>}
        </section>
        <footer className="composer-area">
          <form className="composer" onSubmit={submit}><textarea ref={inputRef} aria-label="Message TARA" value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={onComposerKeyDown} placeholder="Message TARA…  (Shift + Enter for new line)" rows={1} disabled={isBusy} onInput={e => { const el = e.currentTarget; el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 180)}px`; }} />
            <div className="composer-bottom"><div className="composer-tools"><button type="button" className="icon-btn" aria-label="Attachments are not available yet" title="Attachments coming later" onClick={() => { setError("File attachments are not part of the initial MVP."); setRetryAvailable(false); }}><Paperclip size={17} /></button><span className="composer-hint"><ShieldCheck size={13} /> Responses come from your configured TARA backend</span></div><div className="composer-actions">{isBusy && <button type="button" className="stop-btn" onClick={stopRequest} aria-label="Stop generating response"><Square size={13} fill="currentColor" /> Stop</button>}<button type="submit" className="send-btn" disabled={!draft.trim() || isBusy} aria-label="Send message" title={isBusy ? "Wait for the current response" : "Send message"}>{isBusy ? <span className="send-loading" /> : <ArrowUp size={19} />}</button></div></div>
          </form>
          <p className="disclaimer"><ShieldCheck size={12} /> TARA can make mistakes. Verify important information.</p>
        </footer>
      </main>
    </div>
  );
}

export default App;