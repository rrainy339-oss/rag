const els = {
  identityLabel: document.querySelector("#identity-label"),
  logoutButton: document.querySelector("#logout-button"),
  uploadForm: document.querySelector("#upload-form"),
  uploadButton: document.querySelector("#upload-button"),
  uploadStatus: document.querySelector("#upload-status"),
  documentFile: document.querySelector("#document-file"),
  documentTitle: document.querySelector("#document-title"),
  documentTenant: document.querySelector("#document-tenant"),
  documentOwner: document.querySelector("#document-owner"),
  documentGroups: document.querySelector("#document-groups"),
  documentPrincipals: document.querySelector("#document-principals"),
  documentClassification: document.querySelector("#document-classification"),
  refreshDocuments: document.querySelector("#refresh-documents"),
  documentRows: document.querySelector("#document-rows"),
  documentEmpty: document.querySelector("#document-empty"),
  documentsStatus: document.querySelector("#documents-status"),
  editPanel: document.querySelector("#edit-panel"),
  closeEdit: document.querySelector("#close-edit"),
  backdrop: document.querySelector("#backdrop"),
  editTitle: document.querySelector("#edit-title"),
  permissionForm: document.querySelector("#permission-form"),
  permissionDocumentId: document.querySelector("#permission-document-id"),
  permissionTenant: document.querySelector("#permission-tenant"),
  permissionOwner: document.querySelector("#permission-owner"),
  permissionGroups: document.querySelector("#permission-groups"),
  permissionPrincipals: document.querySelector("#permission-principals"),
  permissionClassification: document.querySelector("#permission-classification"),
  permissionStatus: document.querySelector("#permission-status"),
};

const state = {
  me: null,
  documents: [],
  pending: false,
};

init();

async function init() {
  const identity = await RagAuth.requireIdentity({ documents: true });
  if (!identity) return;
  state.me = identity.me;
  if (identity.denied) {
    renderAccessDenied();
    return;
  }
  bindEvents();
  renderIdentity();
  loadDocuments();
}

function bindEvents() {
  els.logoutButton.addEventListener("click", logout);
  els.uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();
    uploadDocument();
  });
  els.refreshDocuments.addEventListener("click", loadDocuments);
  els.closeEdit.addEventListener("click", closeEditPanel);
  els.backdrop.addEventListener("click", closeEditPanel);
  els.permissionForm.addEventListener("submit", (event) => {
    event.preventDefault();
    savePermissions();
  });
}

function renderIdentity() {
  const parts = [state.me?.subject || "管理员"];
  if (state.me?.tenant_id) parts.push(state.me.tenant_id);
  if (state.me?.max_classification) parts.push(`clearance: ${state.me.max_classification}`);
  els.identityLabel.textContent = parts.join(" / ");
}

async function loadDocuments() {
  setDocumentsStatus("正在加载文档...");
  els.refreshDocuments.disabled = true;
  try {
    const data = await RagAuth.requestJson("/api/documents");
    state.documents = Array.isArray(data.documents) ? data.documents : [];
    renderDocuments();
    setDocumentsStatus(state.documents.length ? "" : "暂无文档。");
  } catch (error) {
    setDocumentsStatus(`加载失败：${error.message}`, true);
  } finally {
    els.refreshDocuments.disabled = false;
  }
}

async function uploadDocument() {
  const file = els.documentFile.files?.[0];
  if (!file || state.pending) return;

  state.pending = true;
  els.uploadButton.disabled = true;
  setUploadStatus("正在上传，后端会解析、分块、BGE-M3 编码并写入 Qdrant hybrid。");

  const formData = new FormData();
  formData.append("file", file);
  appendFormValue(formData, "title", els.documentTitle.value);
  appendFormValue(formData, "tenant_id", els.documentTenant.value);
  appendFormValue(formData, "owner_id", els.documentOwner.value);
  appendFormValue(formData, "group_ids", els.documentGroups.value);
  appendFormValue(formData, "principal_ids", els.documentPrincipals.value);
  appendFormValue(formData, "classification", els.documentClassification.value);

  try {
    const response = await RagAuth.request("/api/documents", {
      method: "POST",
      body: formData,
    });
    const documentRecord = await response.json();
    upsertDocument(documentRecord);
    els.uploadForm.reset();
    els.documentClassification.value = "internal";
    renderDocuments();
    setUploadStatus("上传已受理，后台正在处理。");
    scheduleRefresh();
  } catch (error) {
    setUploadStatus(`上传失败：${error.message}`, true);
  } finally {
    state.pending = false;
    els.uploadButton.disabled = false;
  }
}

async function savePermissions() {
  const documentId = els.permissionDocumentId.value;
  if (!documentId) return;
  setPermissionStatus("正在保存权限...");
  try {
    const record = await RagAuth.requestJson(`/api/documents/${encodeURIComponent(documentId)}/permissions`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tenant_id: emptyToNull(els.permissionTenant.value),
        owner_id: emptyToNull(els.permissionOwner.value),
        group_ids: RagAuth.splitList(els.permissionGroups.value),
        principal_ids: RagAuth.splitList(els.permissionPrincipals.value),
        classification: emptyToNull(els.permissionClassification.value),
      }),
    });
    upsertDocument(record);
    renderDocuments();
    setPermissionStatus("权限已保存。");
    scheduleRefresh();
  } catch (error) {
    setPermissionStatus(`保存失败：${error.message}`, true);
  }
}

async function reindexDocument(documentId) {
  setDocumentsStatus("已提交重建索引任务...");
  try {
    const record = await RagAuth.requestJson(`/api/documents/${encodeURIComponent(documentId)}/reindex`, {
      method: "POST",
    });
    upsertDocument(record);
    renderDocuments();
    scheduleRefresh();
  } catch (error) {
    setDocumentsStatus(`重建失败：${error.message}`, true);
  }
}

async function deleteDocument(documentId) {
  const record = state.documents.find((item) => item.document_id === documentId);
  const title = record?.title || record?.filename || documentId;
  if (!window.confirm(`确认删除“${title}”？删除会同步清理 Qdrant 中的向量 points。`)) return;

  setDocumentsStatus("正在删除文档和 Qdrant points...");
  try {
    await RagAuth.requestJson(`/api/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
    });
    state.documents = state.documents.filter((item) => item.document_id !== documentId);
    renderDocuments();
    setDocumentsStatus("文档已删除。");
  } catch (error) {
    setDocumentsStatus(`删除失败：${error.message}`, true);
  }
}

function renderDocuments() {
  els.documentRows.replaceChildren(...state.documents.map(renderDocumentRow));
  els.documentEmpty.classList.toggle("hidden", state.documents.length > 0);
}

function renderDocumentRow(record) {
  const row = document.createElement("tr");

  const documentCell = document.createElement("td");
  documentCell.innerHTML = `<strong>${RagAuth.escapeHtml(record.title || record.filename || "Untitled")}</strong><span>${RagAuth.escapeHtml(record.filename || "")}</span>`;

  const statusCell = document.createElement("td");
  const status = document.createElement("span");
  status.className = `status-pill ${record.status || "uploaded"}`;
  status.textContent = record.status || "uploaded";
  statusCell.append(status);
  if (record.error_message) {
    const error = document.createElement("span");
    error.className = "row-error";
    error.textContent = record.error_message;
    statusCell.append(error);
  }

  const permissionsCell = document.createElement("td");
  permissionsCell.append(...permissionLabels(record).map(chip));

  const indexCell = document.createElement("td");
  indexCell.innerHTML = `<span>${Number(record.chunk_count || 0)} chunks</span><span>${Number(record.indexed_count || 0)} indexed</span>`;

  const updatedCell = document.createElement("td");
  updatedCell.textContent = formatDate(record.updated_at || record.created_at);

  const actionsCell = document.createElement("td");
  actionsCell.className = "row-actions";
  actionsCell.append(
    actionButton("编辑权限", () => openEditPanel(record)),
    actionButton("重建索引", () => reindexDocument(record.document_id)),
    actionButton("删除", () => deleteDocument(record.document_id), "danger"),
  );

  row.append(documentCell, statusCell, permissionsCell, indexCell, updatedCell, actionsCell);
  return row;
}

function openEditPanel(record) {
  els.editTitle.textContent = `编辑权限：${record.title || record.filename || record.document_id}`;
  els.permissionDocumentId.value = record.document_id;
  els.permissionTenant.value = record.tenant_id || "";
  els.permissionOwner.value = record.owner_id || "";
  els.permissionGroups.value = (record.group_ids || []).join(", ");
  els.permissionPrincipals.value = (record.principal_ids || []).join(", ");
  els.permissionClassification.value = record.classification || "";
  setPermissionStatus("");
  els.editPanel.classList.add("open");
  els.backdrop.classList.add("show");
}

function closeEditPanel() {
  els.editPanel.classList.remove("open");
  els.backdrop.classList.remove("show");
}

function permissionLabels(record) {
  const labels = [];
  if (record.tenant_id) labels.push(`tenant: ${record.tenant_id}`);
  if (record.owner_id) labels.push(`owner: ${record.owner_id}`);
  for (const groupId of record.group_ids || []) labels.push(`group: ${groupId}`);
  for (const principalId of record.principal_ids || []) labels.push(`user: ${principalId}`);
  if (record.classification) labels.push(`class: ${record.classification}`);
  return labels.length ? labels : ["public"];
}

function chip(text) {
  const item = document.createElement("span");
  item.className = "permission-chip";
  item.textContent = text;
  return item;
}

function actionButton(text, handler, tone) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `secondary-button compact-button${tone === "danger" ? " danger" : ""}`;
  button.textContent = text;
  button.addEventListener("click", handler);
  return button;
}

function appendFormValue(formData, name, value) {
  const cleaned = String(value || "").trim();
  if (cleaned) formData.append(name, cleaned);
}

function emptyToNull(value) {
  const cleaned = String(value || "").trim();
  return cleaned || null;
}

function upsertDocument(record) {
  const index = state.documents.findIndex((item) => item.document_id === record.document_id);
  if (index >= 0) {
    state.documents.splice(index, 1, record);
  } else {
    state.documents.unshift(record);
  }
}

function scheduleRefresh() {
  window.setTimeout(loadDocuments, 1400);
  window.setTimeout(loadDocuments, 4200);
}

function setUploadStatus(text, isError) {
  els.uploadStatus.textContent = text;
  els.uploadStatus.classList.toggle("error", Boolean(isError));
}

function setDocumentsStatus(text, isError) {
  els.documentsStatus.textContent = text;
  els.documentsStatus.classList.toggle("error", Boolean(isError));
}

function setPermissionStatus(text, isError) {
  els.permissionStatus.textContent = text;
  els.permissionStatus.classList.toggle("error", Boolean(isError));
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function logout() {
  RagAuth.clearSession();
  window.location.href = "./login.html";
}

function renderAccessDenied() {
  document.body.innerHTML = '<main class="login-shell"><section class="login-panel"><h1>无权访问文档管理</h1><a class="primary-button link-action" href="./login.html">重新登录</a></section></main>';
}
