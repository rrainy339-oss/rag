(function () {
  const key = "rag-auth-session-v1";
  const hasSession = Boolean(window.localStorage.getItem(key));
  const target = hasSession ? "./chat.html" : "./login.html";
  window.setTimeout(() => window.location.replace(target), 80);
})();
