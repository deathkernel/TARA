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
  $$(".nav-item").forEach(btn => btn.addEventListener("click", () => {
    if (btn.dataset.view === "tasks" || btn.dataset.view === "tools") toast("This panel is a scaffold; runtime integration is not connected yet.");
    showView(btn.dataset.view);
  }));
  $$("[data-go]").forEach(btn => btn.addEventListener("click", () => showView(btn.dataset.go)));
  $("#top-settings").addEventListener("click", () => showView("settings"));
  $("#open-settings").addEventListener("click", () => showView("settings"));
  $("#runtime-help").addEventListener("click", () => showView("settings"));
  $("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#new-chat").addEventListener("click", () => {
    $("#conversation").innerHTML = '<div class="welcome-note"><span class="note-icon">✦</span><span><strong>New conversation</strong><small>This is a local UI demo; connect a runtime API for model responses.</small></span><button id="dismiss-note" aria-label="Dismiss notice">×</button></div>';
    bindDismiss();
    $("#message-input").value = "";
    $("#message-input").focus();
    showView("chat");
  });
  function bindDismiss() {
    const btn = $("#dismiss-note");
    if (btn) btn.addEventListener("click", () => btn.closest(".welcome-note").remove());
  }
  bindDismiss();
  function appendMessage(text, who) {
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
    row.scrollIntoView({behavior: "smooth", block: "nearest"});
  }
  function submitMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return;
    $("#suggestions").hidden = true;
    const note = $("#conversation .welcome-note");
    if (note) note.remove();
    appendMessage(trimmed, "user");
    appendMessage("The frontend is running, but it is not connected to TARA's inference/runtime API yet. I haven't sent this message to a model. Once a safe local API adapter is implemented, this panel can display real TARA responses.", "assistant");
    $("#message-input").value = "";
    $("#message-input").style.height = "auto";
    toast("Demo message shown. No model request was made.");
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
    $("#conversation").innerHTML = "";
    $("#suggestions").hidden = false;
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