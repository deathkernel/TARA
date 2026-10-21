(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const toastEl = $("#toast");
  let toastTimer;
  const toast = (message) => {
    toastEl.textContent = message;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2800);
  };
  const views = ["chat", "memory", "tasks", "tools", "settings"];
  let conversations = [];
  let activeConversationId = null;
  const historyKey = "tara-chat-history-v1";
  function saveHistory() {
    try { localStorage.setItem(historyKey, JSON.stringify({conversations, activeConversationId})); } catch {}
  }
  function newConversation() {
    const conversation = {id: "chat-" + Date.now() + "-" + Math.random().toString(36).slice(2,7), title: "New chat", messages: []};
    conversations.unshift(conversation);
    activeConversationId = conversation.id;
    saveHistory();
    renderHistory();
    renderConversation();
    showView("chat");
    $("#message-input").focus();
  }
  function renderHistory() {
    const list = $("#recent-list");
    list.replaceChildren();
    if (!conversations.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.textContent = "Your conversations will appear here.";
      list.append(empty);
      return;
    }
    conversations.forEach(conversation => {
      const row = document.createElement("button");
      row.className = "recent-item" + (conversation.id === activeConversationId ? " selected" : "");
      row.title = conversation.title;
      const arrow = document.createElement("span");
      arrow.textContent = "↳";
      const label = document.createElement("span");
      label.textContent = conversation.title;
      const remove = document.createElement("span");
      remove.className = "history-delete";
      remove.textContent = "×";
      remove.setAttribute("role", "button");
      remove.setAttribute("aria-label", "Delete conversation");
      remove.addEventListener("click", event => {
        event.stopPropagation();
        conversations = conversations.filter(c => c.id !== conversation.id);
        if (activeConversationId === conversation.id) activeConversationId = conversations[0]?.id || null;
        saveHistory(); renderHistory(); renderConversation();
      });
      row.append(arrow, label, remove);
      row.addEventListener("click", () => {
        activeConversationId = conversation.id;
        saveHistory(); renderHistory(); renderConversation(); showView("chat");
      });
      list.append(row);
    });
  }
  function showView(name) {
    if (!views.includes(name)) return;
    views.forEach(v => {
      const el = document.getElementById(v + "-view");
      if (el) el.classList.toggle("active-view", v === name);
    });
    $$(".nav-item").forEach(btn => btn.classList.toggle("active", btn.dataset.view === name));
    $("#page-label").textContent = name[0].toUpperCase() + name.slice(1);
    $("#sidebar").classList.remove("open");
  }
  function renderConversation() {
    const conversation = conversations.find(c => c.id === activeConversationId);
    const stage = $(".chat-stage");
    const container = $("#conversation");
    container.replaceChildren();
    stage.classList.toggle("has-messages", !!conversation && conversation.messages.length > 0);
    $("#suggestions").hidden = false;
    if (!conversation || !conversation.messages.length) {
      const note = document.createElement("div");
      note.className = "welcome-note";
      note.innerHTML = '<span class="note-icon">✦</span><span><strong>Start with a question</strong><small>This chat is saved in this browser. AI replies require a connected TARA runtime.</small></span><button id="dismiss-note" aria-label="Dismiss notice">×</button>';
      $("#conversation").append(note);
      bindDismiss();
      return;
    }
    $("#suggestions").hidden = true;
    conversation.messages.forEach(message => appendMessage(message.text, message.role, false));
  }
  $$(".nav-item").forEach(btn => btn.classList.toggle("active", btn.dataset.view === name));
    $("#page-label").textContent = name[0].toUpperCase() + name.slice(1);
    $("#sidebar").classList.remove("open");
  }
  $$(".nav-item").forEach(btn => btn.addEventListener("click", () => {
    if (btn.dataset.view === "tasks" || btn.dataset.view === "tools") toast("This panel is a scaffold; runtime integration is not connected yet.");
    showView(btn.dataset.view);
  }));
  $$("[data-go]").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.go)));
  $("#top-settings").addEventListener("click", () => showView("settings"));
  $("#open-settings").addEventListener("click", () => showView("settings"));
  $("#runtime-help").addEventListener("click", () => showView("settings"));
  $("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#new-chat").addEventListener("click", newConversation);
  function bindDismiss() {
    const btn = $("#dismiss-note");
    if (btn) btn.addEventListener("click", () => btn.closest(".welcome-note").remove());
  }
  bindDismiss();
  try {
    const saved = JSON.parse(localStorage.getItem(historyKey) || "{}");
    conversations = Array.isArray(saved.conversations) ? saved.conversations : [];
    activeConversationId = conversations.some(c => c.id === saved.activeConversationId) ? saved.activeConversationId : (conversations[0]?.id || null);
  } catch { conversations = []; activeConversationId = null; }
  renderHistory();
  renderConversation();
  function appendMessage(text, who, scroll = true) {
    const row = document.createElement("div");
    row.className = "message " + who;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = who === "user" ? "Y" : "T";
    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;
    row.append(avatar, body);
    $("#conversation").append(row);
    if (scroll) row.scrollIntoView({behavior: "smooth", block: "nearest"});
  }
  function submitMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (!activeConversationId) newConversation();
    const conversation = conversations.find(c => c.id === activeConversationId);
    if (!conversation) return;
    conversation.messages.push({role:"user", text:trimmed});
    if (conversation.messages.length === 1) conversation.title = trimmed.length > 32 ? trimmed.slice(0, 29) + "..." : trimmed;
    conversation.messages.push({role:"assistant", text:"The frontend is ready for chat, but TARA's inference API is not connected yet. This is a clearly marked demo response; your message has not been sent to a model."});
    saveHistory(); renderHistory(); renderConversation();
    $("#message-input").value = "";
    $("#message-input").style.height = "auto";
    toast("Conversation saved locally. No model request was made.");
  }
  $("#composer").addEventListener("submit", event => {
    event.preventDefault();
    submitMessage($("#message-input").value);
  });
  $("#message-input").addEventListener("input", event => {
    const el = event.currentTarget;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  });
  $("#message-input").addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  $$("#suggestions button").forEach(btn => btn.addEventListener("click", () => {
    $("#message-input").value = btn.dataset.prompt;
    $("#message-input").focus();
    $("#message-input").dispatchEvent(new Event("input"));
  }));
  $("#attach-button").addEventListener("click", () => toast("Attachments are not implemented in this frontend scaffold."));
  $("#theme-toggle").addEventListener("click", () => {
    document.body.classList.toggle("light");
    toast(document.body.classList.contains("light") ? "Light theme enabled." : "Dark theme enabled.");
  });
  $("#compact-toggle").addEventListener("change", event => {
    document.body.classList.toggle("compact", event.target.checked);
    toast(event.target.checked ? "Compact mode enabled." : "Compact mode disabled.");
  });
  $("#save-url").addEventListener("click", () => {
    const input = $("#api-url");
    try {
      const parsed = new URL(input.value);
      if (!["http:", "https:"].includes(parsed.protocol)) throw new Error("protocol");
      localStorage.setItem("tara-api-base-url", parsed.href.replace(/\/$/, ""));
      $("#url-help").textContent = "Saved locally. This only stores the preference; no connection test was run.";
      toast("API URL saved in this browser only.");
    } catch {
      $("#url-help").textContent = "Enter a valid http:// or https:// URL.";
      toast("That URL doesn't look valid.");
    }
  });
  try {
    const saved = localStorage.getItem("tara-api-base-url");
    if (saved) $("#api-url").value = saved;
  } catch {}
  $("#clear-chat").addEventListener("click", () => {
    if (!window.confirm("Clear the messages currently shown in this conversation?")) return;
    const conversation = conversations.find(c => c.id === activeConversationId);
    if (conversation) conversation.messages = [];
    saveHistory(); renderHistory(); renderConversation();
    toast("Current conversation cleared.");
    showView("chat");
  });
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      $("#new-chat").click();
    }
    if (event.key === "Escape") $("#sidebar").classList.remove("open");
  });
})();