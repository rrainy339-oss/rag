(function () {
  const STORAGE_KEY = "rag-auth-session-v1";
  const DEFAULT_API_BASE = "http://localhost:8000";

  function getSession() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function saveSession(session) {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        ...session,
        apiBase: normalizeApiBase(session.apiBase || DEFAULT_API_BASE),
        savedAt: Date.now(),
      }),
    );
  }

  function clearSession() {
    window.localStorage.removeItem(STORAGE_KEY);
  }

  function apiBase() {
    return normalizeApiBase(getSession()?.apiBase || DEFAULT_API_BASE);
  }

  function apiUrl(path) {
    const cleanPath = String(path || "").startsWith("/") ? path : `/${path}`;
    return `${apiBase()}${cleanPath}`;
  }

  function authHeaders(extraHeaders) {
    const session = getSession();
    const headers = { ...(extraHeaders || {}) };
    if (!session) return headers;

    if (session.authMode === "oidc" && session.bearerToken) {
      headers.Authorization = `Bearer ${session.bearerToken}`;
    }

    if (session.authMode === "dev") {
      setHeader(headers, "x-rag-subject", session.subject);
      setHeader(headers, "x-rag-tenant-id", session.tenantId);
      setHeader(headers, "x-rag-user-id", session.userId);
      setHeader(headers, "x-rag-email", session.email);
      setHeader(headers, "x-rag-group-ids", listValue(session.groupIds));
      setHeader(headers, "x-rag-roles", listValue(session.roles));
      setHeader(headers, "x-rag-scopes", listValue(session.scopes));
      setHeader(headers, "x-rag-max-classification", session.maxClassification);
    }

    return headers;
  }

  async function request(path, options) {
    const init = { ...(options || {}) };
    init.headers = authHeaders(init.headers);
    const response = await window.fetch(apiUrl(path), init);
    if (!response.ok) {
      throw new Error(await errorMessage(response));
    }
    return response;
  }

  async function requestJson(path, options) {
    const response = await request(path, options);
    return response.json();
  }

  async function loadMe() {
    return requestJson("/api/me");
  }

  async function requireIdentity(options) {
    const session = getSession();
    if (!session) {
      redirectToLogin();
      return null;
    }

    let me;
    try {
      me = await loadMe();
    } catch (error) {
      clearSession();
      redirectToLogin(`无法验证登录状态：${error.message}`);
      return null;
    }

    if (options?.documents && !can(me, "rag:documents")) {
      return { me, denied: true };
    }
    if (options?.chat && !can(me, "rag:chat")) {
      return { me, denied: true };
    }
    return { me, denied: false };
  }

  function can(me, scope) {
    if (!me) return false;
    if (!me.enforce_permissions) return true;
    return Array.isArray(me.scopes) && (me.scopes.includes(scope) || me.scopes.includes("rag:admin"));
  }

  function redirectToLogin(message) {
    const target = new URL("./login.html", window.location.href);
    target.searchParams.set("next", window.location.pathname.split("/").pop() || "chat.html");
    if (message) {
      target.searchParams.set("message", message);
    }
    window.location.replace(target.toString());
  }

  function normalizeApiBase(value) {
    return String(value || DEFAULT_API_BASE).trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
  }

  function splitList(value) {
    return String(value || "")
      .replaceAll(";", ",")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function listValue(value) {
    return Array.isArray(value) ? value.join(",") : String(value || "");
  }

  function setHeader(headers, name, value) {
    const cleaned = String(value || "").trim();
    if (cleaned) {
      headers[name] = cleaned;
    }
  }

  async function errorMessage(response) {
    try {
      const payload = await response.json();
      return payload.detail || `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  window.RagAuth = {
    DEFAULT_API_BASE,
    STORAGE_KEY,
    apiBase,
    apiUrl,
    authHeaders,
    can,
    clearSession,
    escapeHtml,
    getSession,
    loadMe,
    request,
    requestJson,
    requireIdentity,
    saveSession,
    splitList,
  };
})();
