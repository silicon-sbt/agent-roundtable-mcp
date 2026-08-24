"""Roundtable's only provider seam: gateway dispatch plus the local mock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from llm_gateway import (
        CLI_BACKENDS as _GATEWAY_CLI_BACKENDS,
        EFFORT_CHOICES,
        TextResponse,
        dispatch_text,
        prompt_to_text,
    )
except ImportError:  # pragma: no cover - llm_gateway is optional; mock works without it.
    _GATEWAY_CLI_BACKENDS = ()
    EFFORT_CHOICES = ()
    TextResponse = None  # type: ignore[assignment,misc]
    dispatch_text = None  # type: ignore[assignment]
    prompt_to_text = None  # type: ignore[assignment]

from .providers_api import MockLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_AVAILABLE = dispatch_text is not None
if GATEWAY_AVAILABLE:
    API_PROVIDERS = (
        "mock",
        "auto",
        "gemini",
        "openai",
        "openrouter",
        "deepseek",
        "dashscope",
        "siliconflow",
        "ollama",
    )
else:
    API_PROVIDERS = ("mock", "auto")
CLI_BACKENDS = _GATEWAY_CLI_BACKENDS
CLI_EFFORT_CHOICES = EFFORT_CHOICES
AUTO_PROVIDER_CHAIN = "gemini,openai,openrouter,mock"


@dataclass(frozen=True)
class ProviderChainResult:
    text: str
    provider: str
    model: str


@dataclass(frozen=True)
class _LocalTextResponse:
    """Minimal TextResponse stand-in used only when llm_gateway is absent."""

    text: str
    provider: str
    model: str
    transport: str = ""
    effort: str = ""


def _prompt_to_text(prompt: Any) -> str:
    """Local fallback for llm_gateway.prompt_to_text when the gateway is absent."""
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(prompt)


def dispatch_provider(
    provider: str,
    prompt: Any,
    *,
    model: str = "",
    effort: str = "",
    service_tier: str = "",
    **kwargs: Any,
) -> Any:
    """Dispatch one route; only the roundtable-owned mock is local."""

    if provider == "mock":
        mock = MockLLM(model=model or "mock")
        response = _LocalTextResponse(
            text=mock.generate(_prompt_to_text(prompt)),
            provider="mock",
            model=mock.model,
            transport="mock",
            effort="",
        )
        if TextResponse is None:
            return response
        return TextResponse(
            response.text,
            provider=response.provider,
            model=response.model,
            transport=response.transport,
            effort=response.effort,
        )
    if dispatch_text is None:
        raise RuntimeError(
            f"Provider '{provider}' requires the optional 'llm_gateway' package. "
            "Install it (pip install -e ../llm_gateway) or use provider='mock'."
        )
    return dispatch_text(
        provider,
        prompt,
        model=model,
        effort=effort,
        service_tier=service_tier,
        project_root=PROJECT_ROOT,
        **kwargs,
    )


__all__ = [
    "API_PROVIDERS",
    "AUTO_PROVIDER_CHAIN",
    "CLI_BACKENDS",
    "CLI_EFFORT_CHOICES",
    "ProviderChainResult",
    "dispatch_provider",
]
