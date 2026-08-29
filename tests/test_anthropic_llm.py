"""Tests for V1 Anthropic (Claude Code) LLM support."""

from __future__ import annotations

from pathlib import Path

from mcp_server.llm_client import AnthropicLLM, resolve_llm

ROOT = Path(__file__).resolve().parent.parent


class _FakeResponse:
    status_code = 200
    text = '{}'
    def json(self):
        return {
            "content": [{"type": "text", "text": "  hello  "}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }


def test_anthropic_generate_uses_messages_api(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr("mcp_server.llm_client.requests.post", fake_post)
    llm = AnthropicLLM(api_key="sk-ant", base_url="https://api.anthropic.com", model="claude")
    out = llm.generate("你好")
    assert out == "hello"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "claude"
    assert captured["json"]["messages"] == [{"role": "user", "content": "你好"}]
    # Anthropic usage: input_tokens / output_tokens mapped to prompt/completion.
    assert llm.last_usage == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_resolve_llm_anthropic_returns_anthropic_client():
    llm = resolve_llm("anthropic", api_key="sk-ant", base_url="https://api.anthropic.com", root_dir=ROOT)
    assert isinstance(llm, AnthropicLLM)
    assert llm.provider_name == "anthropic"


def test_anthropic_in_provider_registry():
    from mcp_server.llm_client import PROVIDER_SPECS, AUTO_ORDER
    assert "anthropic" in PROVIDER_SPECS
    assert "anthropic" in AUTO_ORDER
