RagAuth.configure({
  storageKey: RagAuth.CHAT_STORAGE_KEY,
  loginPage: "./chat-login.html",
});

const STORAGE_KEY = "rag-chat-sessions-v2";
const PROVIDER_DEFAULTS = {
  ollama: { baseUrl: "http://localhost:11434", model: "llama3.1" },
  "openai-compatible": { baseUrl: "https://api.openai.com/v1", model: "gpt-4o-mini" },
};

const icons = {
  trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" /></svg>',
  copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8h11v11H8z" /><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" /></svg>',
};

const els = {
  sessionList: document.querySelector("#session-list"),
  sessionSearch: document.querySelector("#session-search"),
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#empty-state"),
  messageScroll: document.querySelector("#message-scroll"),
  activeTitle: document.querySelector("#active-title"),
  activeSubtitle: document.querySelector("#active-subtitle"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#prompt-input"),
  sendButton: document.querySelector("#send-button"),
  newChat: document.querySelector("#new-chat"),
  settingsPanel: document.querySelector("#settings-panel"),
  toggleSettings: document.querySelector("#toggle-settings"),
  closeSettings: document.querySelector("#close-settings"),
  toggleSidebar: document.querySelector("#toggle-sidebar"),
  sidebar: document.querySelector("#sidebar"),
  backdrop: document.querySelector("#backdrop"),
  identityLabel: document.querySelector("#identity-label"),
  logoutButton: document.querySelector("#logout-button"),
  modeSelect: document.querySelector("#mode-select"),
  llmProvider: document.querySelector("#llm-provider"),
  llmBaseUrl: document.querySelector("#llm-base-url"),
  llmApiKey: document.querySelector("#llm-api-key"),
  llmApiKeyField: document.querySelector("#llm-api-key-field"),
  refreshModels: document.querySelector("#refresh-models"),
  modelOptions: document.querySelector("#model-options"),
  modelsStatus: document.querySelector("#models-status"),
  modelName: document.querySelector("#model-name"),
  topK: document.querySelector("#top-k"),
  temperature: document.querySelector("#temperature"),
  showCitations: document.querySelector("#show-citations"),
};

const state = {
  sessions: loadSessions(),
  activeId: null,
  pending: false,
  me: null,
};

init();

async function init() {
  const identity = await RagAuth.requireIdentity({ chat: true });
  if (!identity) return;
  state.me = identity.me;
  if (identity.denied) {
    renderAccessDenied();
    return;
  }
  if (state.sessions.length === 0) state.sessions.push(createSession());
  state.activeId = state.sessions[0].id;
  bindEvents();
  applyProviderDefaults(false);
  renderIdentity();
  render();
}

function bindEvents() {
  els.newChat.addEventListener("click", () => {
    const session = createSession();
    state.sessions.unshift(session);
    state.activeId = session.id;
    closePanels();
    persist();
    render();
    els.input.focus();
  });
  els.sessionSearch.addEventListener("input", renderSessions);
  els.toggleSettings.addEventListener("click", openSettings);
  els.closeSettings.addEventListener("click", closeSettings);
  els.toggleSidebar.addEventListener("click", openSidebar);
  els.backdrop.addEventListener("click", closePanels);
  els.logoutButton.addEventListener("click", logout);
  els.showCitations.addEventListener("change", renderMessages);
  els.modeSelect.addEventListener("change", renderMessages);
  els.llmProvider.addEventListener("change", () => {
    applyProviderDefaults(true);
    renderMessages();
  });
  els.llmBaseUrl.addEventListener("input", () => setModelsStatus(""));
  els.llmApiKey.addEventListener("input", () => setModelsStatus(""));
  els.refreshModels.addEventListener("click", refreshModels);
  els.modelName.addEventListener("input", renderMessages);
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrompt();
  });
  els.input.addEventListener("input", resizeInput);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitPrompt();
    }
  });
}

function createSession() {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    title: "新会话",
    createdAt: Date.now(),
    messages: [],
  };
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
}

function activeSession() {
  return state.sessions.find((session) => session.id === state.activeId);
}

function renderIdentity() {
  const parts = [state.me?.subject || "用户"];
  if (state.me?.tenant_id) parts.push(state.me.tenant_id);
  els.identityLabel.textContent = parts.join(" / ");
}

function render() {
  renderSessions();
  renderMessages();
  resizeInput();
}

function renderSessions() {
  const query = els.sessionSearch.value.trim().toLowerCase();
  const sessions = state.sessions.filter((session) => session.title.toLowerCase().includes(query));
  els.sessionList.replaceChildren(
    ...sessions.map((session) => {
      const item = document.createElement("button");
      item.className = `session-item${session.id === state.activeId ? " active" : ""}`;
      item.type = "button";
      item.addEventListener("click", () => {
        state.activeId = session.id;
        closePanels();
        render();
      });

      const text = document.createElement("span");
      text.innerHTML = `<span class="session-title">${RagAuth.escapeHtml(session.title)}</span><span class="session-meta">${session.messages.length} 条消息</span>`;

      const remove = document.createElement("span");
      remove.className = "mini-button";
      remove.title = "删除";
      remove.setAttribute("aria-label", "删除");
      remove.innerHTML = icons.trash;
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        removeSession(session.id);
      });

      item.append(text, remove);
      return item;
    }),
  );
}

function renderMessages() {
  const session = activeSession();
  if (!session) return;
  els.activeTitle.textContent = session.title;
  els.activeSubtitle.textContent = els.modeSelect.value === "api" ? `${providerLabel()} / ${els.modelName.value || "模型"}` : "前端模拟";
  els.emptyState.classList.toggle("hidden", session.messages.length > 0);
  els.messages.replaceChildren(...session.messages.map(renderMessage));
  requestAnimationFrame(() => {
    els.messageScroll.scrollTop = els.messageScroll.scrollHeight;
  });
}

function renderMessage(message) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${message.role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = message.role === "user" ? "我" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (message.pending) {
    bubble.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  } else {
    bubble.append(...paragraphs(message.content).map((text) => {
      const p = document.createElement("p");
      p.textContent = text;
      return p;
    }));
    if (message.role === "assistant") bubble.append(renderBubbleActions(message.content));
    if (message.citations?.length && els.showCitations.checked) {
      bubble.append(renderCitations(message.citations));
    }
  }
  wrapper.append(avatar, bubble);
  return wrapper;
}

function renderBubbleActions(content) {
  const actions = document.createElement("div");
  actions.className = "bubble-actions";
  const copy = document.createElement("button");
  copy.className = "mini-button";
  copy.type = "button";
  copy.title = "复制";
  copy.setAttribute("aria-label", "复制");
  copy.innerHTML = icons.copy;
  copy.addEventListener("click", () => navigator.clipboard?.writeText(content));
  actions.append(copy);
  return actions;
}

function renderCitations(citations) {
  const list = document.createElement("div");
  list.className = "citations";
  for (const citation of citations) {
    const item = document.createElement("span");
    item.className = "citation";
    item.innerHTML = `<strong>${RagAuth.escapeHtml(citation.label)}</strong><span>${RagAuth.escapeHtml(citation.text)}</span>`;
    list.append(item);
  }
  return list;
}

function paragraphs(text) {
  return String(text || "").split(/\n{2,}/).map((part) => part.trim()).filter(Boolean);
}

async function submitPrompt() {
  const text = els.input.value.trim();
  if (!text || state.pending) return;
  const session = activeSession();
  if (!session) return;
  if (session.messages.length === 0) {
    session.title = text.length > 22 ? `${text.slice(0, 22)}...` : text;
  }
  session.messages.push({ role: "user", content: text });
  const pending = { role: "assistant", content: "", pending: true };
  session.messages.push(pending);
  els.input.value = "";
  resizeInput();
  state.pending = true;
  els.sendButton.disabled = true;
  persist();
  render();

  try {
    const answer = await generateAnswer(text);
    Object.assign(pending, { pending: false, content: answer.content, citations: answer.citations });
  } catch (error) {
    Object.assign(pending, { pending: false, content: `请求失败：${error.message}`, citations: [] });
  } finally {
    state.pending = false;
    els.sendButton.disabled = false;
    persist();
    render();
  }
}

async function generateAnswer(query) {
  if (els.modeSelect.value === "api") return requestApi(query);
  return mockAnswer(query);
}

async function requestApi(query) {
  const apiKey = els.llmProvider.value === "openai-compatible" ? els.llmApiKey.value.trim() : "";
  const data = await RagAuth.requestJson("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      llm_provider: els.llmProvider.value,
      base_url: els.llmBaseUrl.value.trim(),
      api_key: apiKey || undefined,
      model: els.modelName.value,
      top_k: Number(els.topK.value),
      temperature: Number(els.temperature.value),
    }),
  });
  return {
    content: data.answer || data.content || "",
    citations: normalizeCitations(data.citations || data.contexts || []),
  };
}

async function refreshModels() {
  setModelsStatus("正在获取模型...");
  els.refreshModels.disabled = true;
  try {
    const apiKey = els.llmProvider.value === "openai-compatible" ? els.llmApiKey.value.trim() : "";
    const data = await RagAuth.requestJson("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        llm_provider: els.llmProvider.value,
        base_url: els.llmBaseUrl.value.trim(),
        api_key: apiKey || undefined,
      }),
    });
    const models = Array.isArray(data.models) ? data.models : [];
    renderModelOptions(models);
    if (models.length && shouldReplaceModel()) {
      els.modelName.value = models[0];
      renderMessages();
    }
    setModelsStatus(models.length ? `已获取 ${models.length} 个模型` : "未发现模型");
  } catch (error) {
    setModelsStatus(`获取失败：${error.message}`);
  } finally {
    els.refreshModels.disabled = false;
  }
}

function renderModelOptions(models) {
  els.modelOptions.replaceChildren(...models.map((model) => {
    const option = document.createElement("option");
    option.value = model;
    return option;
  }));
}

function applyProviderDefaults(force) {
  const defaults = PROVIDER_DEFAULTS[els.llmProvider.value] || PROVIDER_DEFAULTS.ollama;
  const knownBaseUrls = Object.values(PROVIDER_DEFAULTS).map((item) => item.baseUrl);
  const knownModels = Object.values(PROVIDER_DEFAULTS).map((item) => item.model).concat("local-model");
  if (force || !els.llmBaseUrl.value.trim() || knownBaseUrls.includes(els.llmBaseUrl.value.trim())) {
    els.llmBaseUrl.value = defaults.baseUrl;
  }
  if (force || !els.modelName.value.trim() || knownModels.includes(els.modelName.value.trim())) {
    els.modelName.value = defaults.model;
  }
  els.llmApiKeyField.classList.toggle("hidden", els.llmProvider.value === "ollama");
  renderModelOptions([]);
  setModelsStatus("");
}

function shouldReplaceModel() {
  const current = els.modelName.value.trim();
  if (!current) return true;
  return Object.values(PROVIDER_DEFAULTS).map((item) => item.model).concat("local-model").includes(current);
}

function providerLabel() {
  return els.llmProvider.value === "ollama" ? "Ollama" : "OpenAI API";
}

function setModelsStatus(text) {
  els.modelsStatus.textContent = text;
}

function mockAnswer(query) {
  return new Promise((resolve) => {
    window.setTimeout(() => resolve({
      content: `已收到问题：“${query}”。\n\n真实模式下，后端会基于当前用户权限进行 Qdrant hybrid 检索，并只使用可访问上下文生成答案。`,
      citations: [{ label: "Context 1", text: "BGE-M3 + Qdrant hybrid" }],
    }), 420);
  });
}

function normalizeCitations(items) {
  return items.slice(0, 8).map((item, index) => ({
    label: item.label || `Context ${index + 1}`,
    text: item.quote || item.text || item.context_id || item.source_chunk_id || item.section_path?.join(" > ") || "citation",
  }));
}

function removeSession(id) {
  const index = state.sessions.findIndex((session) => session.id === id);
  if (index < 0) return;
  state.sessions.splice(index, 1);
  if (state.sessions.length === 0) state.sessions.push(createSession());
  if (state.activeId === id) {
    state.activeId = state.sessions[Math.min(index, state.sessions.length - 1)].id;
  }
  persist();
  render();
}

function resizeInput() {
  els.input.style.height = "0px";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 180)}px`;
}

function openSettings() {
  els.sidebar.classList.remove("open");
  els.settingsPanel.classList.add("open");
  els.backdrop.classList.add("show");
}

function closeSettings() {
  els.settingsPanel.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function openSidebar() {
  els.settingsPanel.classList.remove("open");
  els.sidebar.classList.add("open");
  els.backdrop.classList.add("show");
}

function closePanels() {
  els.settingsPanel.classList.remove("open");
  els.sidebar.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function logout() {
  RagAuth.clearSession();
  window.location.href = "./chat-login.html";
}

function renderAccessDenied() {
  document.body.innerHTML = '<main class="login-shell"><section class="login-panel"><h1>无权访问问答页面</h1><a class="primary-button link-action" href="./chat-login.html">重新登录</a></section></main>';
}
