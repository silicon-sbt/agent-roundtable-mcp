import json
from pathlib import Path

import pytest

from roundtable.agent_llm_config import (
    load_agent_llm_config_document,
    load_agent_llm_configs,
    save_agent_llm_config_document,
)


def test_save_agent_llm_config_document_writes_sanitized_agent_configs(tmp_path: Path):
    save_agent_llm_config_document(
        tmp_path,
        {
            "macroeconomics": {
                "provider_chain": "openrouter:model-a",
                "max_attempts": 2,
                "retry_backoff_seconds": 0.5,
            }
        },
    )

    path = tmp_path / "config" / "agent_llms.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["agents"]["macroeconomics"] == {
        "provider_chain": "openrouter:model-a",
        "max_attempts": 2,
        "retry_backoff_seconds": 0.5,
    }
    assert load_agent_llm_configs(tmp_path)["macroeconomics"]["max_attempts"] == 2


def test_load_agent_llm_config_document_returns_default_when_missing(tmp_path: Path):
    payload = load_agent_llm_config_document(tmp_path)

    assert payload["agents"] == {}
    assert "description" in payload


def test_agent_llm_config_accepts_workflow_style_provider_chain(tmp_path: Path):
    save_agent_llm_config_document(
        tmp_path,
        {
            "macroeconomics": {
                "provider_chain": "claude:sonnet@high, gemini:gemini-2.5-flash",
                "temperature": 0.2,
                "max_output_tokens": 8192,
                "skip": ["claude"],
            }
        },
    )

    payload = load_agent_llm_configs(tmp_path)

    assert payload["macroeconomics"] == {
        "provider_chain": "claude:sonnet@high, gemini:gemini-2.5-flash",
        "temperature": 0.2,
        "max_output_tokens": 8192,
        "skip": ["claude"],
    }


@pytest.mark.parametrize("legacy_key", ["api_key_env", "base_url", "timeout"])
def test_agent_llm_config_rejects_transport_overrides(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    with pytest.raises(ValueError, match="Unsupported LLM config keys"):
        save_agent_llm_config_document(
            tmp_path,
            {"macroeconomics": {"provider": "openrouter", legacy_key: "legacy"}},
        )
