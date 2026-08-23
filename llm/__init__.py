"""Roundtable-specific adapters over the standalone ``llm_gateway`` package."""

from __future__ import annotations

from .facade import (
    ProviderChainLLM,
    TextGenerationRequest,
    create_llm,
    create_llm_from_config,
    generate_text,
)
from .providers_api import LLMClient, MockLLM, describe_llm
from .router import (
    API_PROVIDERS,
    AUTO_PROVIDER_CHAIN,
    CLI_BACKENDS,
    CLI_EFFORT_CHOICES,
)

__all__ = [
    "API_PROVIDERS",
    "AUTO_PROVIDER_CHAIN",
    "CLI_BACKENDS",
    "CLI_EFFORT_CHOICES",
    "LLMClient",
    "MockLLM",
    "ProviderChainLLM",
    "TextGenerationRequest",
    "create_llm",
    "create_llm_from_config",
    "describe_llm",
    "generate_text",
]
