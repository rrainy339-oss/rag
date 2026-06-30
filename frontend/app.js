const STORAGE_KEY = "rag-chat-sessions-v1";
const PROVIDER_DEFAULTS = {
  ollama: {
    baseUrl: "http://localhost:11434",
    model: "llama3.1",
  },
  "openai-compatible": {
    baseUrl: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
  },
};

const icons = {
  trash:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14" /></svg>',
  copy:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8h11v11H8z" /><path d="M5 16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" /></svg>',
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
  documentsPanel: document.querySelector("#documents-panel"),
  toggleDocuments: document.querySelector("#toggle-documents"),
  closeDocuments: document.querySelector("#close-documents"),
  refreshDocuments: document.querySelector("#refresh-documents"),
  documentForm: document.querySelector("#document-form"),
  documentFile: document.querySelector("#document-file"),
  documentTitle: document.querySelector("#document-title"),
  documentTenant: document.querySelector("#document-tenant"),
  documentOwner: document.querySelector("#document-owner"),
  documentGroups: document.querySelector("#document-groups"),
  documentPrincipals: document.querySelector("#document-principals"),
  documentClassification: document.querySelector("#document-classification"),
  documentList: document.querySelector("#document-list"),
  documentsStatus: document.querySelector("#documents-status"),
  toggleSidebar: document.querySelector("#toggle-sidebar"),
  sidebar: document.querySelector(".sidebar"),
  backdrop: document.querySelector("#backdrop"),
  modeSelect: document.querySelector("#mode-select"),
  apiUrl: document.querySelector("#api-url"),
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
  documents: [],
  activeId: null,
  pending: false,
  documentsLoaded: false,
  documentsPending: false,
};

init();

function init() {
  if (state.sessions.length === 0) {
    state.sessions.push(createSession());
  }
  state.activeId = state.sessions[0].id;
  bindEvents();
  applyProviderDefaults(false);
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
  els.toggleSettings.addEventListener("click", () => openSettings());
  els.closeSettings.addEventListener("click", () => closeSettings());
  els.toggleDocuments.addEventListener("click", () => openDocuments());
  els.closeDocuments.addEventListener("click", () => closeDocuments());
  els.refreshDocuments.addEventListener("click", () => loadDocuments());
  els.documentForm.addEventListener("submit", (event) => {
    event.preventDefault();
    uploadDocument();
  });
  els.toggleSidebar.addEventListener("click", () => openSidebar());
  els.backdrop.addEventListener("click", () => closePanels());
  els.showCitations.addEventListener("change", renderMessages);
  els.modeSelect.addEventListener("change", renderMessages);
  els.llmProvider.addEventListener("change", () => {
    applyProviderDefaults(true);
    renderMessages();
  });
  els.llmBaseUrl.addEventListener("input", () => setModelsStatus(""));
  els.llmApiKey.addEventListener("input", () => setModelsStatus(""));
  els.refreshModels.addEventListener("click", () => refreshModels());
  els.modelName.addEventListener("input", renderMessages);

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrompt();
  });

  els.input.addEventListener("input", () => resizeInput());
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
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
  } catch {
    return;
  }
}

function activeSession() {
  return state.sessions.find((session) => session.id === state.activeId);
}

function render() {
  renderSessions();
  renderMessages();
  resizeInput();
}

function renderSessions() {
  const query = els.sessionSearch.value.trim().toLowerCase();
  const sessions = state.sessions.filter((session) =>
    session.title.toLowerCase().includes(query),
  );
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
      text.innerHTML = `<span class="session-title">${escapeHtml(session.title)}</span><span class="session-meta">${session.messages.length} 条消息</span>`;

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
  els.activeSubtitle.textContent =
    els.modeSelect.value === "api"
      ? `${providerLabel()} · ${els.modelName.value || "模型"}`
      : "Local Mock";
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
  avatar.textContent = message.role === "user" ? "你" : "AI";

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
    if (message.role === "assistant") {
      bubble.append(renderBubbleActions(message.content));
    }
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
    item.innerHTML = `<strong>${escapeHtml(citation.label)}</strong><span>${escapeHtml(citation.text)}</span>`;
    list.append(item);
  }
  return list;
}

function paragraphs(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);
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
    Object.assign(pending, {
      pending: false,
      content: answer.content,
      citations: answer.citations,
    });
  } catch (error) {
    Object.assign(pending, {
      pending: false,
      content: `请求失败：${error.message}`,
      citations: [],
    });
  } finally {
    state.pending = false;
    els.sendButton.disabled = false;
    persist();
    render();
  }
}

async function generateAnswer(query) {
  if (els.modeSelect.value === "api") {
    return requestApi(query);
  }
  return mockAnswer(query);
}

async function requestApi(query) {
  const apiKey =
    els.llmProvider.value === "openai-compatible" ? els.llmApiKey.value.trim() : "";
  const response = await fetch(els.apiUrl.value, {
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
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const data = await response.json();
  return {
    content: data.answer || data.content || "",
    citations: normalizeCitations(data.citations || data.contexts || []),
  };
}

async function refreshModels() {
  setModelsStatus("正在获取模型...");
  els.refreshModels.disabled = true;
  try {
    const apiKey =
      els.llmProvider.value === "openai-compatible" ? els.llmApiKey.value.trim() : "";
    const response = await fetch(modelListUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        llm_provider: els.llmProvider.value,
        base_url: els.llmBaseUrl.value.trim(),
        api_key: apiKey || undefined,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
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

async function loadDocuments() {
  setDocumentsStatus("Loading documents...");
  els.refreshDocuments.disabled = true;
  try {
    const response = await fetch(apiEndpoint("documents"));
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    state.documents = Array.isArray(data.documents) ? data.documents : [];
    state.documentsLoaded = true;
    renderDocuments();
    setDocumentsStatus(state.documents.length ? "" : "No documents yet.");
  } catch (error) {
    setDocumentsStatus(`Load failed: ${error.message}`);
  } finally {
    els.refreshDocuments.disabled = false;
  }
}

async function uploadDocument() {
  const file = els.documentFile.files?.[0];
  if (!file || state.documentsPending) return;

  state.documentsPending = true;
  els.documentForm.classList.add("is-pending");
  els.documentForm.querySelector("button[type='submit']").disabled = true;
  setDocumentsStatus("Uploading. Backend will parse, chunk, embed with BGE-M3, and write dense+sparse vectors to Qdrant.");

  const formData = new FormData();
  formData.append("file", file);
  appendFormValue(formData, "title", els.documentTitle.value);
  appendFormValue(formData, "tenant_id", els.documentTenant.value);
  appendFormValue(formData, "owner_id", els.documentOwner.value);
  appendFormValue(formData, "group_ids", els.documentGroups.value);
  appendFormValue(formData, "principal_ids", els.documentPrincipals.value);
  appendFormValue(formData, "classification", els.documentClassification.value);

  try {
    const response = await fetch(apiEndpoint("documents"), {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const documentRecord = await response.json();
    upsertDocument(documentRecord);
    els.documentForm.reset();
    setDocumentsStatus("Upload accepted. Indexing is running in the background.");
    renderDocuments();
    scheduleDocumentRefresh();
  } catch (error) {
    setDocumentsStatus(`Upload failed: ${error.message}`);
  } finally {
    state.documentsPending = false;
    els.documentForm.classList.remove("is-pending");
    els.documentForm.querySelector("button[type='submit']").disabled = false;
  }
}

async function reindexDocument(documentId) {
  setDocumentsStatus("Reindexing document...");
  try {
    const response = await fetch(documentActionUrl(documentId, "reindex"), {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    upsertDocument(await response.json());
    renderDocuments();
    scheduleDocumentRefresh();
  } catch (error) {
    setDocumentsStatus(`Reindex failed: ${error.message}`);
  }
}

async function deleteDocument(documentId) {
  setDocumentsStatus("Deleting document...");
  try {
    const response = await fetch(documentActionUrl(documentId), {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.documents = state.documents.filter((item) => item.document_id !== documentId);
    renderDocuments();
    setDocumentsStatus("");
  } catch (error) {
    setDocumentsStatus(`Delete failed: ${error.message}`);
  }
}

function renderDocuments() {
  if (!state.documentsLoaded && state.documents.length === 0) {
    els.documentList.replaceChildren(emptyDocumentState("Open the panel to load documents."));
    return;
  }
  if (state.documents.length === 0) {
    els.documentList.replaceChildren(emptyDocumentState("No documents uploaded."));
    return;
  }
  els.documentList.replaceChildren(...state.documents.map(renderDocumentCard));
}

function renderDocumentCard(documentRecord) {
  const card = document.createElement("article");
  card.className = "document-card";

  const header = document.createElement("div");
  header.className = "document-card-head";
  const title = document.createElement("div");
  title.className = "document-title";
  title.textContent = documentRecord.title || documentRecord.filename || "Document";
  const status = document.createElement("span");
  status.className = `document-status ${documentRecord.status || "uploaded"}`;
  status.textContent = documentRecord.status || "uploaded";
  header.append(title, status);

  const meta = document.createElement("div");
  meta.className = "document-meta";
  meta.textContent = [
    documentRecord.filename,
    `${documentRecord.chunk_count || 0} chunks`,
    `${documentRecord.indexed_count || 0} indexed`,
  ]
    .filter(Boolean)
    .join(" / ");

  const permissions = document.createElement("div");
  permissions.className = "document-permissions";
  permissionLabels(documentRecord).forEach((label) => {
    const chip = document.createElement("span");
    chip.textContent = label;
    permissions.append(chip);
  });

  const actions = document.createElement("div");
  actions.className = "document-actions";
  const reindex = document.createElement("button");
  reindex.type = "button";
  reindex.className = "secondary-button";
  reindex.textContent = "Reindex";
  reindex.addEventListener("click", () => reindexDocument(documentRecord.document_id));
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "secondary-button danger";
  remove.textContent = "Delete";
  remove.addEventListener("click", () => deleteDocument(documentRecord.document_id));
  actions.append(reindex, remove);

  card.append(header, meta, permissions, actions);
  if (documentRecord.error_message) {
    const error = document.createElement("p");
    error.className = "document-error";
    error.textContent = documentRecord.error_message;
    card.append(error);
  }
  return card;
}

function emptyDocumentState(text) {
  const item = document.createElement("div");
  item.className = "document-empty";
  item.textContent = text;
  return item;
}

function permissionLabels(documentRecord) {
  const labels = [];
  if (documentRecord.tenant_id) labels.push(`tenant: ${documentRecord.tenant_id}`);
  if (documentRecord.owner_id) labels.push(`owner: ${documentRecord.owner_id}`);
  for (const groupId of documentRecord.group_ids || []) labels.push(`group: ${groupId}`);
  for (const principalId of documentRecord.principal_ids || []) labels.push(`user: ${principalId}`);
  if (documentRecord.classification) labels.push(`class: ${documentRecord.classification}`);
  return labels.length ? labels : ["access: unrestricted"];
}

function appendFormValue(formData, name, value) {
  const cleaned = String(value || "").trim();
  if (cleaned) {
    formData.append(name, cleaned);
  }
}

function upsertDocument(documentRecord) {
  const index = state.documents.findIndex(
    (item) => item.document_id === documentRecord.document_id,
  );
  if (index >= 0) {
    state.documents.splice(index, 1, documentRecord);
  } else {
    state.documents.unshift(documentRecord);
  }
  state.documentsLoaded = true;
}

function scheduleDocumentRefresh() {
  window.setTimeout(() => loadDocuments(), 1400);
  window.setTimeout(() => loadDocuments(), 4200);
}

function setDocumentsStatus(text) {
  els.documentsStatus.textContent = text;
}

function renderModelOptions(models) {
  els.modelOptions.replaceChildren(
    ...models.map((model) => {
      const option = document.createElement("option");
      option.value = model;
      return option;
    }),
  );
}

function modelListUrl() {
  return apiEndpoint("models");
}

function documentActionUrl(documentId, action) {
  const url = new URL(apiEndpoint("documents"));
  url.pathname = `${url.pathname.replace(/\/$/, "")}/${encodeURIComponent(documentId)}`;
  if (action) {
    url.pathname += `/${action}`;
  }
  return url.toString();
}

function apiEndpoint(resource) {
  const url = new URL(els.apiUrl.value);
  const apiIndex = url.pathname.indexOf("/api/");
  if (apiIndex >= 0) {
    url.pathname = `${url.pathname.slice(0, apiIndex)}/api/${resource}`;
  } else {
    url.pathname = `/api/${resource}`;
  }
  url.search = "";
  return url.toString();
}

function applyProviderDefaults(force) {
  const defaults = PROVIDER_DEFAULTS[els.llmProvider.value] || PROVIDER_DEFAULTS.ollama;
  const knownBaseUrls = Object.values(PROVIDER_DEFAULTS).map((item) => item.baseUrl);
  const knownModels = [
    ...Object.values(PROVIDER_DEFAULTS).map((item) => item.model),
    "local-model",
  ];

  if (
    force ||
    !els.llmBaseUrl.value.trim() ||
    knownBaseUrls.includes(els.llmBaseUrl.value.trim())
  ) {
    els.llmBaseUrl.value = defaults.baseUrl;
  }
  if (
    force ||
    !els.modelName.value.trim() ||
    knownModels.includes(els.modelName.value.trim())
  ) {
    els.modelName.value = defaults.model;
  }
  els.llmApiKeyField.classList.toggle("hidden", els.llmProvider.value === "ollama");
  renderModelOptions([]);
  setModelsStatus("");
}

function shouldReplaceModel() {
  const current = els.modelName.value.trim();
  if (!current) return true;
  return Object.values(PROVIDER_DEFAULTS)
    .map((item) => item.model)
    .concat("local-model")
    .includes(current);
}

function providerLabel() {
  return els.llmProvider.value === "ollama" ? "Ollama" : "OpenAI API";
}

function setModelsStatus(text) {
  els.modelsStatus.textContent = text;
}

function mockAnswer(query) {
  const citations = [
    {
      label: "Context 1",
      text: "Qdrant Hybrid / BGE-M3",
    },
    {
      label: "Context 2",
      text: "Reranker / Answer Layer",
    },
  ];
  const content = [
    `已收到问题：“${query}”。`,
    "相关答案会基于检索上下文组织，并在关键结论后保留引用 [Context 1]。",
  ].join("\n\n");
  return new Promise((resolve) => {
    setTimeout(() => resolve({ content, citations }), 520);
  });
}

function normalizeCitations(items) {
  return items.slice(0, 8).map((item, index) => ({
    label: item.label || `Context ${index + 1}`,
    text:
      item.quote ||
      item.text ||
      item.context_id ||
      item.source_chunk_id ||
      item.section_path?.join(" > ") ||
      "citation",
  }));
}

function removeSession(id) {
  const index = state.sessions.findIndex((session) => session.id === id);
  if (index < 0) return;
  state.sessions.splice(index, 1);
  if (state.sessions.length === 0) {
    state.sessions.push(createSession());
  }
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
  els.documentsPanel.classList.remove("open");
  els.settingsPanel.classList.add("open");
  els.backdrop.classList.add("show");
}

function closeSettings() {
  els.settingsPanel.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function openDocuments() {
  els.settingsPanel.classList.remove("open");
  els.sidebar.classList.remove("open");
  els.documentsPanel.classList.add("open");
  els.backdrop.classList.add("show");
  if (!state.documentsLoaded) {
    loadDocuments();
  }
}

function closeDocuments() {
  els.documentsPanel.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function openSidebar() {
  els.documentsPanel.classList.remove("open");
  els.sidebar.classList.add("open");
  els.backdrop.classList.add("show");
}

function closePanels() {
  els.settingsPanel.classList.remove("open");
  els.documentsPanel.classList.remove("open");
  els.sidebar.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
