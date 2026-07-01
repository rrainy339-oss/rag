const els = {
  form: document.querySelector("#login-form"),
  apiBase: document.querySelector("#api-base"),
  authMode: document.querySelector("#auth-mode"),
  devFields: document.querySelector("#dev-fields"),
  oidcFields: document.querySelector("#oidc-fields"),
  subject: document.querySelector("#subject"),
  tenantId: document.querySelector("#tenant-id"),
  userId: document.querySelector("#user-id"),
  email: document.querySelector("#email"),
  groupIds: document.querySelector("#group-ids"),
  roles: document.querySelector("#roles"),
  scopes: document.querySelector("#scopes"),
  maxClassification: document.querySelector("#max-classification"),
  bearerToken: document.querySelector("#bearer-token"),
  status: document.querySelector("#login-status"),
};

init();

function init() {
  const session = RagAuth.getSession();
  const message = new URL(window.location.href).searchParams.get("message");
  if (session) {
    els.apiBase.value = session.apiBase || RagAuth.DEFAULT_API_BASE;
    els.authMode.value = session.authMode || "dev";
    els.subject.value = session.subject || "admin-a";
    els.tenantId.value = session.tenantId || "";
    els.userId.value = session.userId || "";
    els.email.value = session.email || "";
    els.groupIds.value = (session.groupIds || []).join(",");
    els.roles.value = (session.roles || []).join(",");
    els.scopes.value = (session.scopes || []).join(",");
    els.maxClassification.value = session.maxClassification || "";
    els.bearerToken.value = session.bearerToken || "";
  }
  if (message) setStatus(message, true);
  toggleAuthFields();
  els.authMode.addEventListener("change", toggleAuthFields);
  els.form.addEventListener("submit", handleSubmit);
}

async function handleSubmit(event) {
  event.preventDefault();
  setStatus("正在验证身份...");

  const session = buildSession();
  RagAuth.saveSession(session);

  try {
    const me = await RagAuth.loadMe();
    setStatus(`登录成功：${me.subject}`);
    window.location.href = nextPage(me);
  } catch (error) {
    RagAuth.clearSession();
    setStatus(`登录失败：${error.message}`, true);
  }
}

function buildSession() {
  const authMode = els.authMode.value;
  return {
    apiBase: els.apiBase.value,
    authMode,
    subject: els.subject.value.trim(),
    tenantId: els.tenantId.value.trim(),
    userId: els.userId.value.trim(),
    email: els.email.value.trim(),
    groupIds: RagAuth.splitList(els.groupIds.value),
    roles: RagAuth.splitList(els.roles.value),
    scopes: RagAuth.splitList(els.scopes.value),
    maxClassification: els.maxClassification.value,
    bearerToken: els.bearerToken.value.trim(),
  };
}

function nextPage(me) {
  const params = new URL(window.location.href).searchParams;
  const requested = params.get("next");
  if (requested && ["chat.html", "admin.html"].includes(requested)) {
    if (requested === "admin.html" && !RagAuth.can(me, "rag:documents")) {
      return "./chat.html";
    }
    return `./${requested}`;
  }
  return RagAuth.can(me, "rag:documents") ? "./admin.html" : "./chat.html";
}

function toggleAuthFields() {
  const mode = els.authMode.value;
  els.devFields.classList.toggle("hidden", mode !== "dev");
  els.oidcFields.classList.toggle("hidden", mode !== "oidc");
}

function setStatus(text, isError) {
  els.status.textContent = text;
  els.status.classList.toggle("error", Boolean(isError));
}
