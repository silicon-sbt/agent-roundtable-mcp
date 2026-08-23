from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from llm_gateway.routing import (
    canonical_backend,
    normalize_effort,
    resolve_transport,
)

from . import transport_cli, transport_http
from .providers_api import MockLLM, create_llm
from .transport_http import antigravity_model_variant


CLI_BACKENDS = (
    "claude",
    "codex",
    "codex_proxy",
    "grok",
    "antigravity",
    "copilot",
)
CLI_EFFORT_BACKENDS = CLI_BACKENDS
CLI_EFFORT_CHOICES = {
    "claude": ("", "low", "medium", "high", "xhigh", "max"),
    "codex": ("", "low", "medium", "high", "xhigh", "max", "ultra"),
    "codex_proxy": ("", "low", "medium", "high", "xhigh", "max"),
    "grok": ("", "low", "medium", "high", "xhigh", "max"),
    "antigravity": ("", "low", "medium", "high", "thinking"),
    "copilot": ("", "none", "low", "medium", "high", "xhigh", "max"),
}
API_PROVIDERS = ("mock", "auto", "gemini", "openai", "openrouter", "deepseek")


@dataclass(frozen=True)
class ProviderChainResult:
    text: str
    provider: str
    model: str


def _alias(value: str) -> str:
    token = (value or "").strip().lower()
    normalized = canonical_backend(token)
    if normalized == "default" and token:
        # Fail closed for ambiguous tokens such as codex_sdk / claude_http,
        # while preserving unrelated direct-API provider names.
        resolve_transport(token)
    return (token or "auto") if normalized == "default" else normalized


def _normalize_effort(value: str) -> str:
    return normalize_effort(value)


def backend_transport(backend: str) -> str:
    return resolve_transport(backend)


def supports_effort(backend: str) -> bool:
    return _alias(backend) in CLI_EFFORT_BACKENDS


def run_cli(
    backend: str,
    text: str,
    model: str = "",
    effort: str = "",
    *,
    env_override: Optional[dict[str, str]] = None,
) -> str:
    normalized = _alias(backend)
    transport = backend_transport(normalized)
    if transport == "subprocess":
        return transport_cli.run_cli(normalized, text, model, effort, env_override=env_override)
    return transport_http.run_cli(normalized, text, model, effort, env_override=env_override)


def split_model_effort(token: str) -> tuple[str, str]:
    model, _, effort = str(token or "").partition("@")
    return model.strip(), _normalize_effort(effort)


def parse_provider_chain(provider_string: str, *, default: str = "gemini") -> list[tuple[str, str]]:
    chain: list[tuple[str, str]] = []
    for segment in str(provider_string or "").split(","):
        segment = segment.strip()
        if not segment:
            continue
        provider, _, model = segment.partition(":")
        raw_provider = provider.strip()
        if not model and "@" in raw_provider:
            raw_provider, _, effort = raw_provider.partition("@")
            model = f"@{effort.strip()}"
        provider = _alias(raw_provider)
        chain.append((provider, model.strip()))
    return chain or [(_alias(default), "")]


def cli_text(
    backend: str,
    prompt: Any,
    *,
    workflow: str = "",
    model: Optional[str] = None,
    effort: Optional[str] = None,
) -> str:
    normalized = _alias(backend)
    if normalized not in CLI_BACKENDS:
        raise ValueError(f"cli_text only supports {', '.join(CLI_BACKENDS)}, got {backend!r}.")
    chosen, inline_effort = split_model_effort(model or "")
    chosen_effort = _normalize_effort(effort or inline_effort or "")
    if not supports_effort(normalized):
        chosen_effort = ""
    text = prompt if isinstance(prompt, str) else str(prompt)
    return run_cli(normalized, text, chosen, chosen_effort)


def call_provider(
    instruction: str,
    provider: str,
    model: str = "",
    *,
    workflow: str,
    fallback_model: str = "",
    gemini_max_output_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    normalized = _alias(provider)
    chosen, inline_effort = split_model_effort(model or fallback_model)
    if normalized == "mock":
        return MockLLM(model=chosen or "mock").generate(instruction)
    if normalized in CLI_BACKENDS:
        return cli_text(
            normalized,
            instruction,
            workflow=workflow,
            model=chosen,
            effort=inline_effort or None,
        )
    llm = create_llm(
        normalized,
        model=chosen or None,
        temperature=temperature,
        max_output_tokens=gemini_max_output_tokens,
    )
    return llm.generate(instruction)


def run_provider_chain(
    instruction: str,
    provider_string: str,
    *,
    workflow: str,
    skip: tuple[str, ...] = (),
    label: str = "",
    validate: Optional[Callable[[str], Any]] = None,
    **call_kwargs: Any,
) -> tuple[str, str]:
    result = run_provider_chain_detailed(
        instruction,
        provider_string,
        workflow=workflow,
        skip=skip,
        label=label,
        validate=validate,
        **call_kwargs,
    )
    return result.text, result.provider


def run_provider_chain_detailed(
    instruction: str,
    provider_string: str,
    *,
    workflow: str,
    skip: tuple[str, ...] = (),
    label: str = "",
    validate: Optional[Callable[[str], Any]] = None,
    **call_kwargs: Any,
) -> ProviderChainResult:
    errors: list[str] = []
    tag = label or workflow
    skip_set = {_alias(item) for item in skip}
    for provider, model in parse_provider_chain(provider_string):
        if provider in skip_set:
            continue
        try:
            result = call_provider(
                instruction,
                provider,
                model,
                workflow=workflow,
                **call_kwargs,
            )
            if validate is not None:
                validate(result)
            chosen, _effort = split_model_effort(
                model or str(call_kwargs.get("fallback_model", ""))
            )
            return ProviderChainResult(
                text=result,
                provider=provider,
                model=chosen or "default",
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")
            print(f"  [{tag}] provider={provider} failed, trying next provider... ({exc})", flush=True)
    raise RuntimeError("; ".join(errors) or "No usable provider in chain")


def model_for_provider(provider_chain: str, provider: str) -> str:
    normalized = _alias(provider)
    for chain_provider, model in parse_provider_chain(provider_chain):
        if chain_provider == normalized:
            chosen, _ = split_model_effort(model)
            return chosen or "default"
    return "unknown"
