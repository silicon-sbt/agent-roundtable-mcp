from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from llm_gateway.routing import resolve_transport

from llm import (
    DeepSeekLLM,
    MockLLM,
    OpenAILLM,
    OpenRouterLLM,
    TextGenerationRequest,
    create_llm_from_config,
    generate_text,
    parse_provider_chain,
    split_model_effort,
)
from llm import environment, router, transport_cli, transport_http
from llm.transport_http import antigravity_model_variant
from roundtable.loader import PROJECT_ROOT


def test_parse_provider_chain_accepts_workflow_style_model_tokens():
    assert parse_provider_chain(
        "antigravity:gemini-3.1-pro-low, claude:sonnet@high, "
        "codex@medium, codex_proxy:gpt-5.6-terra@high"
    ) == [
        ("antigravity", "gemini-3.1-pro-low"),
        ("claude", "sonnet@high"),
        ("codex", "@medium"),
        ("codex_proxy", "gpt-5.6-terra@high"),
    ]
    assert split_model_effort("sonnet@high") == ("sonnet", "high")
    assert split_model_effort("@medium") == ("", "medium")


def test_generate_text_uses_provider_chain_fallback(monkeypatch):
    calls: list[tuple[str, str, str, str]] = []

    def fake_call_provider(
        instruction: str,
        provider: str,
        model: str = "",
        *,
        workflow: str,
        **_: object,
    ) -> str:
        calls.append((instruction, provider, model, workflow))
        if provider == "claude":
            raise RuntimeError("quota exhausted")
        return f"{provider}:{model}:{instruction}"

    monkeypatch.setattr("llm.router.call_provider", fake_call_provider)

    text, provider = generate_text(
        TextGenerationRequest(
            workflow="agent_roundtable",
            provider_chain="claude:sonnet@high, gemini:gemini-2.5-flash",
            label="test-chain",
        ),
        "hello",
    )

    assert provider == "gemini"
    assert text == "gemini:gemini-2.5-flash:hello"
    assert calls == [
        ("hello", "claude", "sonnet@high", "agent_roundtable"),
        ("hello", "gemini", "gemini-2.5-flash", "agent_roundtable"),
    ]


def test_create_llm_from_provider_chain_config_uses_facade_client():
    llm = create_llm_from_config(
        {
            "provider_chain": "mock:chain-model",
            "temperature": 0.2,
            "max_output_tokens": 2048,
        },
        fallback=MockLLM(model="fallback"),
    )

    assert llm.generate("[ROUND_TABLE_FINAL_SUMMARY]\n话题：测试") != ""
    assert getattr(llm, "provider_name") == "mock"
    assert getattr(llm, "model") == "chain-model"


def test_provider_chain_llm_records_the_model_that_actually_succeeded(monkeypatch):
    def fake_call_provider(instruction, provider, model="", **_kwargs):
        if model == "first-model":
            raise RuntimeError("first model unavailable")
        return f"{provider}:{model}:{instruction}"

    monkeypatch.setattr(router, "call_provider", fake_call_provider)
    llm = create_llm_from_config(
        {"provider_chain": "gemini:first-model, gemini:second-model"}
    )

    assert llm.generate("hello") == "gemini:second-model:hello"
    assert getattr(llm, "provider_name") == "gemini"
    assert getattr(llm, "model") == "second-model"


def test_cli_transport_uses_repo_root_and_does_not_mutate_parent_env(monkeypatch):
    captured = {}

    def fake_run_local_cli(
        backend,
        text,
        model,
        effort,
        *,
        project_root,
        env_override,
    ):
        captured.update(
            backend=backend,
            text=text,
            model=model,
            effort=effort,
            project_root=project_root,
            env_override=env_override,
        )
        return "ok"

    monkeypatch.delenv("ROUNDTRIP_TEST_ENV", raising=False)
    monkeypatch.setattr(transport_cli, "run_local_cli", fake_run_local_cli)

    result = transport_cli.run_cli(
        "claude",
        "hello",
        "sonnet",
        "high",
        env_override={"ROUNDTRIP_TEST_ENV": "child-only"},
    )

    assert result == "ok"
    assert transport_cli.PROJECT_ROOT == PROJECT_ROOT
    assert captured["project_root"] == PROJECT_ROOT
    assert captured["env_override"]["ROUNDTRIP_TEST_ENV"] == "child-only"
    assert "ROUNDTRIP_TEST_ENV" not in os.environ


def test_project_environment_loads_optional_shared_env_after_local_env(tmp_path, monkeypatch):
    shared_env = tmp_path / "shared.env"
    shared_env.write_text(
        "SHARED_ENV_ONLY=shared\n"
        "SHARED_ENV_BLANK_LOCAL=shared\n"
        "ENV_PRECEDENCE_TEST=shared\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        f"LLM_SHARED_ENV_FILE={shared_env}\n"
        "SHARED_ENV_BLANK_LOCAL=\n"
        "ENV_PRECEDENCE_TEST=project\n",
        encoding="utf-8",
    )
    names = (
        "LLM_SHARED_ENV_FILE",
        "SHARED_ENV_ONLY",
        "SHARED_ENV_BLANK_LOCAL",
        "ENV_PRECEDENCE_TEST",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment, "_DOTENV_LOADED", False)

    try:
        environment.load_project_env()
        assert os.environ["SHARED_ENV_ONLY"] == "shared"
        assert os.environ["SHARED_ENV_BLANK_LOCAL"] == "shared"
        assert os.environ["ENV_PRECEDENCE_TEST"] == "project"
    finally:
        for name in names:
            os.environ.pop(name, None)


def test_local_and_proxy_transport_boundaries_cannot_be_overridden(monkeypatch):
    monkeypatch.setenv("CLAUDE_TRANSPORT", "http")
    monkeypatch.setenv("CODEX_TRANSPORT", "http")
    monkeypatch.setenv("CODEX_PROXY_TRANSPORT", "subprocess")
    monkeypatch.setenv("CLI_TRANSPORT", "sdk")

    assert router.backend_transport("claude") == "subprocess"
    assert router.backend_transport("codex") == "subprocess"
    assert router.backend_transport("codex_proxy") == "http"
    assert router.backend_transport("antigravity") == "http"
    assert router.backend_transport("grok") == "http"
    assert router.backend_transport("copilot") == "http"

    with pytest.raises(ValueError, match="本地官方 CLI"):
        resolve_transport("claude", "http")
    with pytest.raises(ValueError, match="CLIProxyAPI HTTP"):
        resolve_transport("codex_proxy", "subprocess")
    with pytest.raises(ValueError, match="混淆本地订阅与代理身份"):
        router.backend_transport("codex_sdk")


def test_current_effort_catalog_supports_new_codex_levels_and_flat_variants():
    assert router.CLI_EFFORT_CHOICES["codex"][-2:] == ("max", "ultra")
    assert (
        antigravity_model_variant("gemini-3.5-flash-extra-low", "high")
        == "gemini-3.5-flash-extra-low"
    )


def test_run_cli_dispatches_local_and_proxy_identities_without_fallback(monkeypatch):
    calls = []

    def fake_local(backend, text, model, effort, *, env_override=None):
        calls.append(("local", backend, text, model, effort, env_override))
        return "local-ok"

    def fake_proxy(backend, text, model, effort, *, env_override=None):
        calls.append(("http", backend, text, model, effort, env_override))
        return "http-ok"

    monkeypatch.setattr(router.transport_cli, "run_cli", fake_local)
    monkeypatch.setattr(router.transport_http, "run_cli", fake_proxy)

    local_result = router.run_cli(
        "codex",
        "hello",
        "gpt-test",
        "high",
        env_override={"TEST_ONLY": "1"},
    )
    proxy_result = router.run_cli(
        "codex_proxy",
        "proxy hello",
        "gpt-proxy",
        "medium",
        env_override={"TEST_ONLY": "2"},
    )

    assert local_result == "local-ok"
    assert proxy_result == "http-ok"
    assert calls == [
        ("local", "codex", "hello", "gpt-test", "high", {"TEST_ONLY": "1"}),
        ("http", "codex_proxy", "proxy hello", "gpt-proxy", "medium", {"TEST_ONLY": "2"}),
    ]


def test_transport_adapters_fail_closed_before_any_external_call():
    with pytest.raises(RuntimeError, match="只接受 claude/codex"):
        transport_cli.run_cli("codex_proxy", "hello")
    with pytest.raises(RuntimeError, match="禁止发送到 CLIProxyAPI"):
        transport_http.run_cli("codex", "hello")


def test_openai_compatible_clients_forward_max_output_tokens(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    clients = [
        OpenAILLM(api_key="test", max_output_tokens=111),
        OpenRouterLLM(api_key="test", model="router-model", max_output_tokens=222),
        DeepSeekLLM(api_key="test", max_output_tokens=333),
    ]
    for client in clients:
        assert client.generate("hello") == "ok"

    assert [call["max_tokens"] for call in calls] == [111, 222, 333]
