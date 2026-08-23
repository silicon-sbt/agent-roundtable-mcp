from __future__ import annotations

from pathlib import Path

from llm_gateway import example_model_ids, read_model_example


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_EXAMPLE = PROJECT_ROOT / "config" / "model_example.json"


def test_generated_model_example_is_the_offline_catalog() -> None:
    payload = read_model_example(MODEL_EXAMPLE)
    assert payload["generator"] == "llm-gateway refresh-example"
    assert payload["refresh_command"].startswith("llm-gateway refresh-example")
    assert set(payload["providers"]) >= {
        "claude",
        "codex",
        "codex_proxy",
        "antigravity",
        "grok",
        "gemini",
        "openai",
        "openrouter",
        "deepseek",
    }


def test_example_model_ids_reads_structured_provider_sections() -> None:
    payload = read_model_example(MODEL_EXAMPLE)
    codex_models = example_model_ids(payload, "codex")
    proxy_models = example_model_ids(payload, "codex_proxy")
    assert "gpt-5.6-sol" in codex_models
    assert "gpt-5.6-terra" in proxy_models
    assert all("://" not in model for model in codex_models + proxy_models)


def test_copy_paste_chains_use_structured_routes() -> None:
    payload = read_model_example(MODEL_EXAMPLE)
    chains = payload["_copy_paste_chains"]
    assert chains
    for routes in chains.values():
        assert routes
        assert all(isinstance(route, dict) for route in routes)
        assert all(route.get("provider") and route.get("model") for route in routes)


def test_ui_uses_gateway_schema_helpers_instead_of_recursive_string_scraping() -> None:
    source = (PROJECT_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert "read_model_example" in source
    assert "example_model_ids" in source
    assert "list_provider_model_ids" in source
    assert "_flatten_provider_chain_presets" not in source
    assert "model_catalog.json" not in source
