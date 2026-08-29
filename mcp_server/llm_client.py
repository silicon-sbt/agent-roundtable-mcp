"""LLM client resolution for the MCP roundtable runner.

The MCP service can run real roundtables without the optional llm_gateway
package: any OpenAI-compatible chat endpoint works (DeepSeek / OpenAI /
OpenRouter / Gemini / custom base_url). Keys are read from .env.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from llm.providers_api import LLMClient, MockLLM

# ---------------------------------------------------------------------------
# Minimal .env loader (no third-party dependency)
# ---------------------------------------------------------------------------

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ENV_LOADED: set[Path] = set()


def load_env_files(root_dir: Path | str) -> None:
    """Load .env files into the process environment (idempotent)."""
    root = Path(root_dir)
    path = (root / ".env").resolve()
    if path in _ENV_LOADED or not path.exists():
        return
    _ENV_LOADED.add(path)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "\'"):
            value = value[1:-1]
        if value and key not in os.environ:
            os.environ[key] = value


def load_agent_credential(root_dir: Path | str | None = None) -> bool:
    """Detect an OpenAI-compatible credential so the MCP aligns with the MCP client.

    Alignment means the roundtable uses the SAME api as the requesting agent.
    Detection order (client-agnostic):
      1. Provider keys already in the client env (DeepSeek / OpenAI / OpenRouter /
         Gemini) - covers clients like Claude Code / Cursor that export their key to
         the MCP subprocess. The repo's own .env key is ignored (it is not the
         agent's key).
      2. DSH's own credential (~/.dsh/.credentials.yaml, provider deepseek-official).

    Returns True when an OpenAI-compatible api key is available for the roundtable;
    else False, and a real run should ask the caller to pass api_key/base_url.
    """
    client_key_envs = [
        "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2",
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
        "ANTHROPIC_API_KEY",
    ]
    # Snapshot the client env BEFORE loading the repo .env, so a repo .env key
    # (the roundtable's own, not the agent's) never counts as "aligned".
    client_keys = {name: os.getenv(name, "").strip() for name in client_key_envs}
    if root_dir is not None:
        load_env_files(root_dir)
    for name in AUTO_ORDER:
        spec = PROVIDER_SPECS[name]
        if any(client_keys.get(k) for k in spec.api_key_envs):
            return True
    path = Path(os.path.expanduser("~/.dsh/.credentials.yaml"))
    if not path.exists():
        return False
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    match = re.search(r"DEEPSEEK_API_KEY\s*:\s*['\"]?([A-Za-z0-9._\-]{12,})", raw)
    if not match:
        return False
    os.environ["DEEPSEEK_API_KEY"] = match.group(1)
    return True

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_envs: tuple[str, ...]
    default_model: str


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        name="deepseek",
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key_envs=("DEEPSEEK_API_KEY",),
        default_model="deepseek-chat",
    ),
    "openai": ProviderSpec(
        name="openai",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key_envs=("OPENAI_API_KEY",),
        default_model="gpt-4o-mini",
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key_envs=("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"),
        default_model="openai/gpt-4o-mini",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        base_url=os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        api_key_envs=("GEMINI_API_KEY_1", "GEMINI_API_KEY_2"),
        default_model="gemini-2.0-flash",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        api_key_envs=("ANTHROPIC_API_KEY",),
        default_model="claude-sonnet-4-20250514",
    ),
}

AUTO_ORDER = ("deepseek", "openai", "openrouter", "gemini", "anthropic")


def _first_key(spec: ProviderSpec) -> str | None:
    for env_name in spec.api_key_envs:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAICompatLLM:
    """Satisfies the roundtable LLMClient protocol (generate / provider_name / model)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str = "openai-compatible",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        # Objective token counter refreshed on every successful generate(); used by
        # the V2 L2 audit (never trust the agent's self-report for token usage).
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str) -> str:
        """Call the chat endpoint with a small retry budget for transient failures."""
        url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        attempts = 0
        while True:
            attempts += 1
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempts >= self.max_retries:
                    raise RuntimeError("LLM request to " + url + " failed: " + str(exc)) from exc
                continue
            if response.status_code != 200:
                if (
                    attempts < self.max_retries
                    and response.status_code in (429, 500, 502, 503, 504)
                ):
                    continue
                message = "LLM API error %s from %s: %s" % (
                    response.status_code, self.provider_name, response.text[:300],
                )
                raise RuntimeError(message)
            try:
                data = response.json()
                usage = data.get("usage") or {}
                self.last_usage = {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                }
                return str(data["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                message = "Unexpected LLM API response from %s: %s" % (
                    self.provider_name, response.text[:300],
                )
                raise RuntimeError(message) from exc



class AnthropicLLM:
    """Anthropic Messages API client (Claude Code / Claude native).

    Not OpenAI-compatible: /v1/messages, x-api-key + anthropic-version headers, and
    input_tokens/output_tokens usage. Exposes the same LLMClient protocol so the
    roundtable can align with Claude Code.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str = "anthropic",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        timeout: float = 180.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str) -> str:
        url = self.base_url + "/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        attempts = 0
        while True:
            attempts += 1
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempts >= self.max_retries:
                    raise RuntimeError("LLM request to " + url + " failed: " + str(exc)) from exc
                continue
            if response.status_code != 200:
                if attempts < self.max_retries and response.status_code in (429, 500, 502, 503, 504):
                    continue
                raise RuntimeError("LLM API error %s from %s: %s" % (response.status_code, self.provider_name, response.text[:300]))
            try:
                data = response.json()
                usage = data.get("usage") or {}
                imp = int(usage.get("input_tokens", 0))
                out = int(usage.get("output_tokens", 0))
                self.last_usage = {"prompt_tokens": imp, "completion_tokens": out, "total_tokens": imp + out}
                text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
                return text.strip()
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise RuntimeError("Unexpected LLM API response from %s: %s" % (self.provider_name, response.text[:300])) from exc


def _build_client(
    spec: ProviderSpec,
    key: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> LLMClient:
    """Anthropic is not OpenAI-compatible; dispatch the right client."""
    if spec.name == "anthropic":
        return AnthropicLLM(
            api_key=key, base_url=base_url, model=model, provider_name=spec.name,
            temperature=temperature, max_output_tokens=max_output_tokens,
        )
    return OpenAICompatLLM(
        api_key=key, base_url=base_url, model=model, provider_name=spec.name,
        temperature=temperature, max_output_tokens=max_output_tokens,
    )


def resolve_llm(
    provider: str = "auto",
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 4096,
    root_dir: Path | str | None = None,
) -> LLMClient:
    """Build an LLM client; mock and no-key auto fall back to MockLLM."""
    if root_dir is not None:
        load_env_files(root_dir)
    selected = (provider or "auto").strip().lower()

    if selected == "mock":
        return MockLLM(model=model or "mock")

    if selected == "auto":
        for name in AUTO_ORDER:
            spec = PROVIDER_SPECS[name]
            key = api_key or _first_key(spec)
            if key:
                return _build_client(
                    spec, key,
                    base_url=base_url or spec.base_url,
                    model=model or spec.default_model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
        return MockLLM(model="mock")

    spec = PROVIDER_SPECS.get(selected)
    if spec is not None:
        key = api_key or _first_key(spec)
        if not key:
            env_names = "/".join(spec.api_key_envs)
            raise ValueError(
                "Provider " + repr(selected) + " needs a key. Set " + env_names
                + " in .env or pass api_key.",
            )
        return _build_client(
            spec, key,
            base_url=base_url or spec.base_url,
            model=model or spec.default_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    # Custom OpenAI-compatible endpoint.
    if not base_url or not api_key:
        raise ValueError(
            "Unknown provider " + repr(selected)
            + ". Use mock, auto, one of " + str(sorted(PROVIDER_SPECS))
            + ", or provide base_url + api_key for a custom endpoint.",
        )
    return OpenAICompatLLM(
        api_key=api_key,
        base_url=base_url,
        model=model or "gpt-4o-mini",
        provider_name=selected,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def provider_status(root_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Describe configured providers (used by the provider_info tool)."""
    if root_dir is not None:
        load_env_files(root_dir)
    status: list[dict[str, Any]] = []
    for name in AUTO_ORDER:
        spec = PROVIDER_SPECS[name]
        status.append(
            {
                "name": name,
                "configured": _first_key(spec) is not None,
                "default_model": spec.default_model,
                "base_url": spec.base_url,
            }
        )
    return status


__all__ = [
    "OpenAICompatLLM",
    "AnthropicLLM",
    "ProviderSpec",
    "PROVIDER_SPECS",
    "AUTO_ORDER",
    "load_agent_credential",
    "load_env_files",
    "provider_status",
    "resolve_llm",
]
