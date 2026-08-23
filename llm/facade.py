from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import router
from .providers_api import LLMClient, create_llm


@dataclass(frozen=True)
class TextGenerationRequest:
    workflow: str
    provider_chain: str
    fallback_model: str = ""
    temperature: float = 0.3
    max_output_tokens: int = 4096
    label: str = ""
    skip: tuple[str, ...] = ()


def generate_text(
    request: TextGenerationRequest,
    prompt: str,
    validate: Callable[[str], Any] | None = None,
) -> tuple[str, str]:
    result = generate_text_detailed(request, prompt, validate)
    return result.text, result.provider


def generate_text_detailed(
    request: TextGenerationRequest,
    prompt: str,
    validate: Callable[[str], Any] | None = None,
) -> router.ProviderChainResult:
    return router.run_provider_chain_detailed(
        prompt,
        request.provider_chain,
        workflow=request.workflow,
        skip=request.skip,
        label=request.label,
        validate=validate,
        fallback_model=request.fallback_model,
        gemini_max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
    )


class ProviderChainLLM:
    def __init__(self, request: TextGenerationRequest):
        self.request = request
        self.provider_name = "provider_chain"
        self.model = request.provider_chain

    def generate(self, prompt: str) -> str:
        result = generate_text_detailed(self.request, prompt)
        self.provider_name = result.provider
        self.model = result.model
        return result.text


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value),)


def create_llm_from_config(config: dict[str, Any] | None, fallback: LLMClient | None = None) -> LLMClient:
    if not config:
        if fallback is not None:
            return fallback
        return create_llm("auto")

    provider_chain = str(config.get("provider_chain") or "").strip()
    if provider_chain:
        return ProviderChainLLM(
            TextGenerationRequest(
                workflow=str(config.get("workflow") or "agent_roundtable"),
                provider_chain=provider_chain,
                fallback_model=str(config.get("fallback_model") or ""),
                temperature=float(config.get("temperature", 0.3)),
                max_output_tokens=int(config.get("max_output_tokens", 4096)),
                label=str(config.get("label") or ""),
                skip=_tuple(config.get("skip")),
            )
        )

    provider = str(config.get("provider", "auto"))
    if provider == "inherit":
        if fallback is None:
            return create_llm("auto")
        return fallback

    timeout = config.get("timeout")
    return create_llm(
        provider=provider,
        model=config.get("model"),
        api_key_env=config.get("api_key_env"),
        base_url=config.get("base_url"),
        timeout=int(timeout) if timeout is not None else None,
        temperature=float(config.get("temperature", 0.7)),
        max_output_tokens=(
            int(config["max_output_tokens"]) if config.get("max_output_tokens") is not None else None
        ),
    )
