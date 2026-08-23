from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .environment import load_project_env


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class LLMConfigurationError(RuntimeError):
    """Raised when a provider is selected without required configuration."""


class GeminiAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Gemini HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _extract(prompt: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}：(.+)", prompt)
    return match.group(1).strip() if match else ""


@dataclass
class MockLLM:
    """Deterministic local LLM for tests and no-key demos."""

    model: str = "mock"
    provider_name: str = "mock"

    def generate(self, prompt: str) -> str:
        topic = _extract(prompt, "话题") or "这个话题"
        if "[ROUND_TABLE_MODERATOR_QUESTION]" in prompt:
            round_text = _extract(prompt, "当前轮次") or "本轮"
            return (
                f"{round_text} 我们先把问题收窄：围绕“{topic}”，请各位说明最关键的判断依据，"
                "并直接回应上一位的逻辑漏洞或盲点。"
            )
        if "[ROUND_TABLE_ROUND_SUMMARY]" in prompt:
            return (
                "1. 本轮把讨论从直觉判断推进到因果链拆解。\n"
                "2. 主要争议在于短期冲击和长期适应能力谁更重要。\n"
                "3. 下一轮应继续追问哪些假设最容易被现实推翻。"
            )
        if "[ROUND_TABLE_FINAL_SUMMARY]" in prompt:
            return (
                "## 主要共识\n"
                "- 这个问题不能用单一变量解释，需要同时看技术、制度、周期和人性。\n\n"
                "## 主要分歧\n"
                "- 分歧集中在风险出现的速度、可控性，以及普通人能否及时调整。\n\n"
                "## 最有价值观点\n"
                "- 最值得保留的是把宏大判断拆成可验证假设，而不是只做情绪化站队。\n\n"
                "## 风险提示\n"
                "- 涉及现实经济或市场数据的判断需要联网核实，不能把模拟讨论当成事实来源。\n\n"
                "## 最终总结\n"
                f"- 圆桌倾向于认为“{topic}”存在真实变量和不确定性，行动上应保持开放、审慎和可调整。"
            )

        persona_name = _extract(prompt, "- 名称") or "Agent"
        role = _extract(prompt, "- 角色") or "观察者"
        catchphrases = _extract(prompt, "- 口头禅")
        phrase = catchphrases.split(",")[0].strip() if catchphrases else "先拆开看"
        return (
            f"{phrase}。我作为{role}，会先回应上一位：他的观点有价值，但容易把“{topic}”看成单线故事。"
            "更稳妥的做法是把它拆成几个可检验假设：谁受益、谁承担成本、变化速度有多快、制度能否缓冲。"
            "如果有本地参考资料，我会把它当作背景线索而不是绝对答案；如果涉及现实经济数据，需要联网核实。"
            "在没有核实时，我只讨论逻辑结构而不报具体数字。"
            f"这是 {persona_name} 的基本判断。"
        )


def _load_environment() -> None:
    load_project_env()


def collect_numbered_keys(prefix: str) -> list[str]:
    direct_key = os.getenv(prefix)
    numbered: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name, value in os.environ.items():
        match = pattern.match(name)
        if match and value:
            numbered.append((int(match.group(1)), value))

    keys = [value for _, value in sorted(numbered)]
    if direct_key:
        keys.insert(0, direct_key)
    return keys


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_numbered_api_key_env(prefix: str) -> str:
    numbered: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for name, value in os.environ.items():
        match = pattern.match(name)
        if match and value:
            numbered.append((int(match.group(1)), name))
    if numbered:
        return min(numbered)[1]
    return prefix


@dataclass
class GeminiLLM:
    model: str = "gemini-2.5-flash"
    api_keys: list[str] | None = None
    models: list[str] | None = None
    temperature: float = 0.7
    max_output_tokens: int = 4096
    timeout: int = 60
    provider_name: str = "gemini"

    def __post_init__(self) -> None:
        _load_environment()
        if self.api_keys is None:
            self.api_keys = collect_numbered_keys("GEMINI_API_KEY")
        if not self.api_keys:
            raise LLMConfigurationError(
                "Gemini provider selected but no GEMINI_API_KEY or GEMINI_API_KEY_N was found."
            )
        if self.models is None:
            self.models = _split_csv(self.model)
        if not self.models:
            raise LLMConfigurationError("Gemini provider selected but no model was configured.")

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for model in self.models or []:
            for api_key in self.api_keys or []:
                try:
                    return self._generate_with_key(prompt, api_key, model)
                except GeminiAPIError as exc:
                    last_error = exc
                    if exc.status_code in {400, 404, 503}:
                        break
                    time.sleep(0.2)
                except Exception as exc:  # Try the next free key/model before giving up.
                    last_error = exc
                    time.sleep(0.2)
        raise RuntimeError(f"Gemini request failed for all configured keys: {last_error}") from last_error

    def _generate_with_key(self, prompt: str, api_key: str, model: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GeminiAPIError(exc.code, detail) from exc

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise RuntimeError(f"Gemini returned an empty response: {data}")
        return text


@dataclass
class OpenAILLM:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: int = 60
    temperature: float = 0.7
    max_output_tokens: int = 4096
    provider_name: str = "openai"

    def __post_init__(self) -> None:
        _load_environment()
        self.api_key = self.api_key or os.getenv(self.api_key_env)
        if not self.api_key:
            raise LLMConfigurationError(
                f"OpenAI provider selected but {self.api_key_env} was not found."
            )

    def generate(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content or ""


@dataclass
class OpenRouterLLM:
    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str = "OPENROUTER_API_KEY_1"
    models: list[str] | None = None
    timeout: int = 60
    temperature: float = 0.7
    max_output_tokens: int = 4096
    provider_name: str = "openrouter"

    def __post_init__(self) -> None:
        _load_environment()
        self.api_key = self.api_key or os.getenv(self.api_key_env)
        if not self.api_key and self.api_key_env == "OPENROUTER_API_KEY_1":
            legacy_key = os.getenv("OPENROUTER_API_KEY")
            if legacy_key:
                self.api_key = legacy_key
                self.api_key_env = "OPENROUTER_API_KEY"
        if not self.api_key:
            raise LLMConfigurationError(
                f"OpenRouter provider selected but {self.api_key_env} was not found."
            )
        if self.models is None:
            self.models = _split_csv(self.model)
        if not self.models:
            raise LLMConfigurationError("OpenRouter provider selected but no model was configured.")

    def generate(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=self.timeout,
        )
        last_error: Exception | None = None
        for model in self.models or []:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"OpenRouter request failed for all configured models: {last_error}") from last_error


@dataclass
class DeepSeekLLM:
    model: str = "deepseek-v4-flash"
    api_key: str | None = None
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    timeout: int = 60
    temperature: float = 0.7
    max_output_tokens: int = 4096
    provider_name: str = "deepseek"

    def __post_init__(self) -> None:
        _load_environment()
        self.api_key = self.api_key or os.getenv(self.api_key_env)
        if not self.api_key:
            raise LLMConfigurationError(
                f"DeepSeek provider selected but {self.api_key_env} was not found."
            )

    def generate(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content or ""


def describe_llm(llm: LLMClient) -> dict[str, str]:
    provider = str(getattr(llm, "provider_name", llm.__class__.__name__))
    models = getattr(llm, "models", None)
    if models:
        model = ",".join(str(item) for item in models)
    else:
        model = str(getattr(llm, "model", "unknown"))
    return {"provider": provider, "model": model}


def _configured_api_key(api_key_env: str | None) -> str | None:
    return os.getenv(api_key_env) if api_key_env else None


def create_llm(
    provider: str = "auto",
    model: str | None = None,
    *,
    api_key_env: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMClient:
    _load_environment()
    selected = provider.lower()
    temp = 0.7 if temperature is None else temperature
    if selected == "mock":
        return MockLLM(model=model or "mock")
    if selected == "auto":
        if collect_numbered_keys("GEMINI_API_KEY"):
            selected = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            selected = "openai"
        elif collect_numbered_keys("OPENROUTER_API_KEY"):
            selected = "openrouter"
        else:
            return MockLLM()

    if selected == "gemini":
        gemini_model = model or os.getenv("GEMINI_MODEL") or os.getenv("GEMINI_MODEL_NAME")
        gemini_api_keys = [_configured_api_key(api_key_env)] if api_key_env else None
        return GeminiLLM(
            model=gemini_model or "gemini-2.5-flash",
            api_keys=[key for key in gemini_api_keys if key] if gemini_api_keys else None,
            temperature=temp,
            max_output_tokens=max_output_tokens
            or int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
            timeout=timeout or int(os.getenv("GEMINI_TIMEOUT", "60")),
        )
    if selected == "openai":
        return OpenAILLM(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key_env=api_key_env or "OPENAI_API_KEY",
            timeout=timeout or int(os.getenv("OPENAI_TIMEOUT", "60")),
            temperature=temp,
            max_output_tokens=max_output_tokens
            or int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096")),
        )
    if selected == "openrouter":
        return OpenRouterLLM(
            model=model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            api_key_env=api_key_env or _default_numbered_api_key_env("OPENROUTER_API_KEY"),
            timeout=timeout or int(os.getenv("OPENROUTER_TIMEOUT", "60")),
            temperature=temp,
            max_output_tokens=max_output_tokens
            or int(os.getenv("OPENROUTER_MAX_OUTPUT_TOKENS", "4096")),
        )
    if selected == "deepseek":
        return DeepSeekLLM(
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            api_key_env=api_key_env or "DEEPSEEK_API_KEY",
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout or int(os.getenv("DEEPSEEK_TIMEOUT", "60")),
            temperature=temp,
            max_output_tokens=max_output_tokens
            or int(os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "4096")),
        )
    raise ValueError(f"Unsupported provider: {provider}")
