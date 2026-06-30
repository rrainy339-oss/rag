from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    metadata: dict[str, object] = field(default_factory=dict)


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, prompt: str) -> LLMResponse:
        """Generate an answer from a prompt."""


class MockLLMProvider:
    provider_name = "mock"

    def __init__(self, response_text: str, *, model_name: str = "mock-llm") -> None:
        self.response_text = response_text
        self.model_name = model_name

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.response_text,
            provider=self.provider_name,
            model=self.model_name,
            metadata={"prompt_chars": len(prompt)},
        )


class OllamaProvider:
    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_name: str,
        temperature: float = 0.1,
        top_p: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

    def generate(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer strictly from supplied context and cite it.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        parsed = _request_json(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
            error_prefix="Ollama request failed",
        )

        try:
            text = parsed["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError("Ollama response did not match chat format.") from exc

        return LLMResponse(
            text=str(text).strip(),
            provider=self.provider_name,
            model=self.model_name,
            metadata={
                "done_reason": parsed.get("done_reason"),
                "eval_count": parsed.get("eval_count"),
                "prompt_eval_count": parsed.get("prompt_eval_count"),
            },
        )


class OpenAICompatibleProvider:
    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer strictly from supplied context and cite it.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        parsed = _request_json(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=self._headers(),
            timeout=self.timeout,
            error_prefix="LLM request failed",
        )

        try:
            text = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("LLM response did not match OpenAI chat format.") from exc

        return LLMResponse(
            text=str(text).strip(),
            provider=self.provider_name,
            model=self.model_name,
            metadata={
                "usage": parsed.get("usage"),
                "finish_reason": _finish_reason(parsed),
            },
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def list_ollama_models(
    *,
    base_url: str = "http://localhost:11434",
    timeout: float = 30.0,
) -> list[str]:
    parsed = _request_json(
        f"{base_url.rstrip('/')}/api/tags",
        timeout=timeout,
        error_prefix="Ollama model list request failed",
    )
    models = parsed.get("models")
    if not isinstance(models, list):
        raise LLMProviderError("Ollama model list response did not match format.")
    names = [
        str(item.get("name") or item.get("model")).strip()
        for item in models
        if isinstance(item, dict) and (item.get("name") or item.get("model"))
    ]
    return sorted(name for name in names if name)


def list_openai_compatible_models(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[str]:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    parsed = _request_json(
        f"{base_url.rstrip('/')}/models",
        headers=headers,
        timeout=timeout,
        error_prefix="OpenAI-compatible model list request failed",
    )
    data = parsed.get("data")
    if not isinstance(data, list):
        raise LLMProviderError(
            "OpenAI-compatible model list response did not match format."
        )
    names = [
        str(item.get("id")).strip()
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]
    return sorted(name for name in names if name)


def _request_json(
    url: str,
    *,
    data: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
    error_prefix: str,
) -> dict[str, object]:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        headers=headers or {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise LLMProviderError(f"{error_prefix} with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMProviderError(f"{error_prefix}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMProviderError(f"{error_prefix} timed out.") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"{error_prefix}: response was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError(f"{error_prefix}: response JSON was not an object.")
    return parsed


def _finish_reason(payload: dict[str, object]) -> object | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return first.get("finish_reason")
