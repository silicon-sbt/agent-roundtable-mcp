"""Roundtable's only provider seam: gateway dispatch plus the local mock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_gateway import (
    CLI_BACKENDS,
    EFFORT_CHOICES,
    TextResponse,
    dispatch_text,
    prompt_to_text,
)

from .providers_api import MockLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
CLI_EFFORT_CHOICES = EFFORT_CHOICES
AUTO_PROVIDER_CHAIN = "gemini,openai,openrouter,mock"


@dataclass(frozen=True)
class ProviderChainResult:
    text: str
    provider: str
    model: str


def dispatch_provider(
    provider: str,
    prompt: Any,
    *,
    model: str = "",
    effort: str = "",
    service_tier: str = "",
    **kwargs: Any,
) -> TextResponse:
    """Dispatch one route; only the roundtable-owned mock is local."""

    if provider == "mock":
        mock = MockLLM(model=model or "mock")
        return TextResponse(
            mock.generate(prompt_to_text(prompt)),
            provider="mock",
            model=mock.model,
            transport="mock",
            effort="",
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
