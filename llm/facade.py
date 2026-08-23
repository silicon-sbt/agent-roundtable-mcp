"""Roundtable object facade backed by the shared gateway chain executor."""

from __future__ import annotations

from typing import Any, Callable

from llm_gateway import ChainEvent, ProviderRoute, TextGenerationRequest
from llm_gateway import generate_text as gateway_generate_text

from . import router
from .providers_api import LLMClient, MockLLM


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
    """Execute the gateway chain while retaining effective model metadata."""

    events: list[ChainEvent] = []
    text, provider = gateway_generate_text(
        request,
        prompt,
        validate,
        dispatch=router.dispatch_provider,
        on_event=events.append,
    )
    success = next((event for event in reversed(events) if event.status == "success"), None)
    model = success.model if success is not None else "unknown"
    return router.ProviderChainResult(text=text, provider=provider, model=model)


class ProviderChainLLM:
    """Adapt a gateway provider chain to the roundtable's ``generate`` protocol."""

    def __init__(self, request: TextGenerationRequest):
        self.request = request
        self.provider_name = "provider_chain"
        self.model = str(request.provider_chain)

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


def create_llm(
    provider: str = "auto",
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> LLMClient:
    """Create a roundtable LLM without reimplementing provider clients."""

    selected = str(provider or "auto").strip().lower()
    if selected == "mock":
        return MockLLM(model=model or "mock")
    provider_chain: str | list[ProviderRoute]
    if selected == "auto":
        provider_chain = router.AUTO_PROVIDER_CHAIN
    else:
        provider_chain = [ProviderRoute(selected, model or "")]
    return ProviderChainLLM(
        TextGenerationRequest(
            workflow="agent_roundtable",
            provider_chain=provider_chain,
            temperature=0.7 if temperature is None else temperature,
            max_output_tokens=max_output_tokens or 4096,
        )
    )


def create_llm_from_config(
    config: dict[str, Any] | None,
    fallback: LLMClient | None = None,
) -> LLMClient:
    """Bind project configuration to the gateway's provider-chain request."""

    if not config:
        return fallback if fallback is not None else create_llm("auto")

    forbidden = {"api_key_env", "base_url", "timeout"}.intersection(config)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise ValueError(
            f"Transport overrides are owned by llm_gateway and cannot appear here: {names}"
        )

    provider_chain = config.get("provider_chain")
    if provider_chain:
        return ProviderChainLLM(
            TextGenerationRequest(
                workflow=str(config.get("workflow") or "agent_roundtable"),
                provider_chain=provider_chain,
                fallback_model=str(config.get("fallback_model") or ""),
                temperature=float(config.get("temperature", 0.3)),
                max_output_tokens=int(config.get("max_output_tokens", 4096)),
                gemini_rotation_strategy=str(config.get("gemini_rotation_strategy") or "key"),
                max_attempts=int(config.get("max_attempts", 1)),
                retry_backoff_seconds=float(config.get("retry_backoff_seconds", 0.0)),
                label=str(config.get("label") or ""),
                skip=_tuple(config.get("skip")),
            )
        )

    provider = str(config.get("provider", "auto"))
    if provider == "inherit":
        return fallback if fallback is not None else create_llm("auto")
    return create_llm(
        provider=provider,
        model=str(config.get("model") or "") or None,
        temperature=float(config.get("temperature", 0.7)),
        max_output_tokens=(
            int(config["max_output_tokens"])
            if config.get("max_output_tokens") is not None
            else None
        ),
    )


__all__ = [
    "ProviderChainLLM",
    "TextGenerationRequest",
    "create_llm",
    "create_llm_from_config",
    "generate_text",
    "generate_text_detailed",
]
