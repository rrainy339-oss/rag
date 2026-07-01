(function () {
  const ADMIN_STORAGE_KEY = "rag-admin-auth-session-v1";
  const CHAT_STORAGE_KEY = "rag-chat-auth-session-v1";
  const DEFAULT_API_BASE = "http://localhost:8000";

  const state = {
    storageKey: CHAT_STORAGE_KEY,
    loginPage: "./chat-login.html",
  };

  function configure(options) {
    state.storageKey = options?.storageKey || state.storageKey;
    state.loginPage = options?.loginPage || state.loginPage;
  }

  function getSession() {
    try {
      const raw = window.localStorage.getItem(state.storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function saveSession(session) {
    window.localStorage.setItem(
      state.storageKey,
      JSON.stringify({
        ...session,
        apiBase: normalizeApiBase(session.apiBase || DEFAULT_API_BASE),
        savedAt: Date.now(),
      }),
    );
  }

  function clearSession() {
    window.localStorage.removeItem(state.storageKey);
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
    if (session?.accessToken) {
      headers.Authorization = `Bearer ${session.accessToken}`;
    }
    return headers;
  }

  async function login(kind, credentials) {
    const apiBaseValue = normalizeApiBase(credentials.apiBase || DEFAULT_API_BASE);
    const response = await window.fetch(`${apiBaseValue}/api/auth/${kind}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: credentials.username,
        password: credentials.password,
      }),
    });
    if (!response.ok) {
      throw new Error(await errorMessage(response));
    }
    const token = await response.json();
    saveSession({
      apiBase: apiBaseValue,
      accessToken: token.access_token,
      tokenType: token.token_type || "bearer",
      expiresIn: token.expires_in,
      principal: token.principal,
      loginKind: kind,
    });
    return token;
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
    if (!session?.accessToken) {
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
    return (
      Array.isArray(me.scopes) &&
      (me.scopes.includes(scope) || me.scopes.includes("rag:admin"))
    );
  }

  function redirectToLogin(message) {
    const target = new URL(state.loginPage, window.location.href);
    const currentPage = window.location.pathname.split("/").pop();
    if (currentPage) {
      target.searchParams.set("next", currentPage);
    }
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
    ADMIN_STORAGE_KEY,
    CHAT_STORAGE_KEY,
    DEFAULT_API_BASE,
    apiBase,
    apiUrl,
    authHeaders,
    can,
    clearSession,
    configure,
    escapeHtml,
    getSession,
    loadMe,
    login,
    request,
    requestJson,
    requireIdentity,
    saveSession,
    splitList,
  };
})();
