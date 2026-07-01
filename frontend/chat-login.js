RagAuth.configure({
  storageKey: RagAuth.CHAT_STORAGE_KEY,
  loginPage: "./chat-login.html",
});

const els = {
  form: document.querySelector("#chat-login-form"),
  apiBase: document.querySelector("#api-base"),
  username: document.querySelector("#username"),
  password: document.querySelector("#password"),
  status: document.querySelector("#login-status"),
};

init();

function init() {
  const session = RagAuth.getSession();
  const message = new URL(window.location.href).searchParams.get("message");
  if (session?.apiBase) {
    els.apiBase.value = session.apiBase;
  }
  if (message) setStatus(message, true);
  els.form.addEventListener("submit", handleSubmit);
}

async function handleSubmit(event) {
  event.preventDefault();
  setStatus("正在登录用户账号...");
  try {
    const token = await RagAuth.login("chat", {
      apiBase: els.apiBase.value,
      username: els.username.value.trim(),
      password: els.password.value,
    });
    if (!RagAuth.can(token.principal, "rag:chat")) {
      RagAuth.clearSession();
      setStatus("该账号没有问答权限。", true);
      return;
    }
    setStatus(`登录成功：${token.principal.subject}`);
    window.location.href = "./chat.html";
  } catch (error) {
    RagAuth.clearSession();
    setStatus(`登录失败：${error.message}`, true);
  }
}

function setStatus(text, isError) {
  els.status.textContent = text;
  els.status.classList.toggle("error", Boolean(isError));
}
