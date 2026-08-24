from __future__ import annotations

from pathlib import Path

import pytest

from llm import (
    MockLLM,
    ProviderChainLLM,
    TextGenerationRequest,
    create_llm,
    create_llm_from_config,
    generate_text,
)
from llm import providers_api, router
pytest.importorskip("llm_gateway")
from llm_gateway import TextResponse


def _response(provider: str, model: str, content: str = "ok") -> TextResponse:
    transport = "subprocess" if provider in {"claude", "codex"} else "direct"
    if provider in {"codex_proxy", "antigravity", "grok"}:
        transport = "http"
    return TextResponse(content, provider, model, transport, "")


def test_roundtable_exports_only_business_provider_objects() -> None:
    assert providers_api.__all__ == ["LLMClient", "MockLLM", "describe_llm"]
    assert not hasattr(providers_api, "OpenRouterLLM")
    assert not hasattr(providers_api, "GeminiLLM")


def test_removed_project_transport_and_catalog_modules_are_absent() -> None:
    package = Path(router.__file__).parent
    for name in ("environment.py", "transport_cli.py", "transport_http.py", "catalog.py"):
        assert not (package / name).exists()
    assert not hasattr(router, "parse_provider_chain")
    assert not hasattr(router, "run_provider_chain")


def test_roundtable_dispatch_seam_forwards_to_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_dispatch(provider: str, prompt: str, **kwargs):
        captured.update(provider=provider, prompt=prompt, **kwargs)
        return _response(provider, kwargs["model"], "shared")

    monkeypatch.setattr(router, "dispatch_text", fake_dispatch)
    result = router.dispatch_provider(
        "codex",
        "hello",
        model="gpt-5.6-sol",
        effort="high",
        temperature=0.2,
        max_output_tokens=123,
    )
    assert result.content == "shared"
    assert captured["provider"] == "codex"
    assert captured["model"] == "gpt-5.6-sol"
    assert captured["effort"] == "high"
    assert captured["temperature"] == 0.2
    assert captured["max_output_tokens"] == 123
    assert captured["project_root"] == router.PROJECT_ROOT


def test_gateway_chain_handles_fallback_and_effective_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_dispatch(provider: str, _prompt: str, **kwargs):
        calls.append(provider)
        if provider == "gemini":
            raise RuntimeError("quota")
        return _response(provider, kwargs["model"], "deep result")

    monkeypatch.setattr(router, "dispatch_text", fake_dispatch)
    llm = create_llm_from_config(
        {"provider_chain": "gemini:flash,deepseek:deepseek-chat"}
    )
    assert llm.generate("prompt") == "deep result"
    assert llm.provider_name == "deepseek"
    assert llm.model == "deepseek-chat"
    assert calls == ["gemini", "deepseek"]


def test_validation_failure_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_dispatch(provider: str, _prompt: str, **kwargs):
        return _response(provider, kwargs["model"], "bad" if provider == "gemini" else "valid")

    monkeypatch.setattr(router, "dispatch_text", fake_dispatch)
    text, provider = generate_text(
        TextGenerationRequest(
            workflow="agent_roundtable",
            provider_chain="gemini:first,deepseek:second",
        ),
        "prompt",
        validate=lambda value: (_ for _ in ()).throw(ValueError("invalid"))
        if value == "bad"
        else None,
    )
    assert (text, provider) == ("valid", "deepseek")


def test_create_llm_keeps_only_mock_and_gateway_chain_objects() -> None:
    assert isinstance(create_llm("mock"), MockLLM)
    direct = create_llm("openrouter", "openrouter/free")
    assert isinstance(direct, ProviderChainLLM)
    assert direct.request.provider_chain[0].provider == "openrouter"
    automatic = create_llm("auto")
    assert isinstance(automatic, ProviderChainLLM)
    assert automatic.request.provider_chain == router.AUTO_PROVIDER_CHAIN


def test_provider_chain_object_updates_effective_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        router,
        "dispatch_text",
        lambda provider, _prompt, **kwargs: _response(provider, kwargs["model"], "answer"),
    )
    llm = create_llm_from_config(
        {
            "provider_chain": "openrouter:openrouter/free",
            "temperature": 0.1,
            "max_output_tokens": 99,
        }
    )
    assert isinstance(llm, ProviderChainLLM)
    assert llm.generate("prompt") == "answer"
    assert llm.provider_name == "openrouter"
    assert llm.model == "openrouter/free"


@pytest.mark.parametrize("key", ["api_key_env", "base_url", "timeout"])
def test_project_config_rejects_transport_overrides(key: str) -> None:
    with pytest.raises(ValueError, match="owned by llm_gateway"):
        create_llm_from_config({"provider": "openrouter", key: "legacy"})


def test_auto_chain_reaches_roundtable_mock_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(provider: str, _prompt: str, **_kwargs):
        raise RuntimeError(f"{provider} unavailable")

    monkeypatch.setattr(router, "dispatch_text", unavailable)
    llm = create_llm("auto")
    assert "观察者" in llm.generate("话题：统一网关")
    assert llm.provider_name == "mock"
    assert llm.model == "mock"
