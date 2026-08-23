from llm.catalog import (
    fetch_model_options,
    fallback_model_options,
    list_api_key_env_names,
)


def test_list_api_key_env_names_prefers_numbered_openrouter_keys():
    environ = {
        "OPENROUTER_API_KEY": "legacy-key",
        "OPENROUTER_API_KEY_2": "key-2",
        "OPENROUTER_API_KEY_1": "key-1",
    }

    assert list_api_key_env_names("openrouter", environ=environ) == [
        "OPENROUTER_API_KEY_1",
        "OPENROUTER_API_KEY_2",
        "OPENROUTER_API_KEY",
    ]


def test_fallback_model_options_combines_current_and_env_models():
    environ = {
        "OPENROUTER_MODEL": "model-b, model-c",
    }

    assert fallback_model_options(
        "openrouter",
        current_model="model-a",
        environ=environ,
    ) == ["model-a", "model-b", "model-c"]


def test_deepseek_model_options_use_official_models_and_ignore_openrouter_current_model():
    environ = {
        "DEEPSEEK_MODEL": "deepseek-v4-pro",
    }

    assert fallback_model_options(
        "deepseek",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        environ=environ,
    ) == [
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek-v4-pro",
    ]


def test_fetch_deepseek_model_options_delegates_to_gateway(monkeypatch):
    monkeypatch.setattr(
        "llm.catalog.list_direct_model_ids",
        lambda provider, **_kwargs: ["deepseek-v4-flash", "deepseek-v4-pro"],
    )

    result = fetch_model_options(
        "deepseek",
        current_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )

    assert result.source == "llm_gateway"
    assert result.error is None
    assert result.models[:2] == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_fetch_claude_models_delegates_to_shared_gateway(monkeypatch):
    monkeypatch.setattr(
        "llm.catalog.list_local_claude_models",
        lambda: ["sonnet", "opus"],
    )

    result = fetch_model_options("claude")

    assert result.models == ["sonnet", "opus"]
    assert result.source == "claude_aliases"
    assert result.error is None


def test_fetch_codex_models_reads_local_cache(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(
        '{"models":[{"slug":"gpt-live","visibility":"list"}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    result = fetch_model_options("codex", root_dir=tmp_path)

    assert result.models == ["gpt-live"]
    assert result.source == "codex_cache"
    assert result.error is None


def test_fetch_cliproxy_models_delegates_to_shared_gateway(tmp_path, monkeypatch):
    captured = {}

    def fake_list(provider, *, project_root, env_override):
        captured.update(
            provider=provider,
            project_root=project_root,
            env_override=env_override,
        )
        return ["gpt-proxy-live"]

    monkeypatch.setattr("llm.catalog.list_proxy_model_ids", fake_list)
    monkeypatch.setenv("CLI_PROXY_BASE_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("CLI_PROXY_API_KEY", "test-only")

    result = fetch_model_options("codex_proxy", root_dir=tmp_path, timeout=7)

    assert result.models == ["gpt-proxy-live"]
    assert result.source == "cliproxy"
    assert captured["provider"] == "codex_proxy"
    assert captured["project_root"] == tmp_path
    assert captured["env_override"]["CLI_PROXY_BASE_URL"] == "http://127.0.0.1:9999"
    assert captured["env_override"]["CLI_PROXY_API_KEY"] == "test-only"
    assert captured["env_override"]["CLI_PROXY_TIMEOUT"] == "7"


def test_fetch_cliproxy_models_can_read_shared_env_file(tmp_path, monkeypatch):
    shared_env = tmp_path / "shared.env"
    shared_env.write_text(
        "CLI_PROXY_BASE_URL=http://127.0.0.1:9777\n"
        "CLI_PROXY_API_KEY=shared-only\n",
        encoding="utf-8",
    )
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(
        f"LLM_SHARED_ENV_FILE={shared_env}\n",
        encoding="utf-8",
    )
    for name in (
        "LLM_SHARED_ENV_FILE",
        "CLI_PROXY_BASE_URL",
        "CLI_PROXY_API_KEY",
        "CLIPROXY_BASE_URL",
        "CLIPROXY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    captured = {}

    def fake_list(provider, *, project_root, env_override):
        captured.update(
            provider=provider,
            project_root=project_root,
            env_override=env_override,
        )
        return ["grok-shared"]

    monkeypatch.setattr("llm.catalog.list_proxy_model_ids", fake_list)

    result = fetch_model_options("grok", root_dir=project_root, timeout=4)

    assert result.models == ["grok-shared"]
    assert result.source == "cliproxy"
    assert captured["provider"] == "grok"
    assert captured["project_root"] == project_root
    assert captured["env_override"]["CLI_PROXY_BASE_URL"] == "http://127.0.0.1:9777"
    assert captured["env_override"]["CLI_PROXY_API_KEY"] == "shared-only"


def test_fetch_cliproxy_error_keeps_source_label(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llm.catalog.list_proxy_model_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("proxy unavailable")),
    )

    result = fetch_model_options("codex_proxy", root_dir=tmp_path)

    assert result.source == "cliproxy"
    assert "proxy unavailable" in (result.error or "")
