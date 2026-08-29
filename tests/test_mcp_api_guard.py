"""V1 MCP run_roundtable API-alignment guard tests."""

from __future__ import annotations

from pathlib import Path

from mcp_server.core import run_roundtable_impl
from mcp_server.llm_client import MockLLM

ROOT = Path(__file__).resolve().parent.parent


def test_real_run_without_caller_api_asks_for_key(tmp_path, monkeypatch):
    """When neither the caller's api nor DSH's credential is available, a real
    roundtable must ask for the api instead of silently using a mismatched key."""
    monkeypatch.setattr("mcp_server.core.load_agent_credential", lambda *a, **k: False)
    result = run_roundtable_impl(
        "API 对齐测试",
        council="experts",
        rounds=1,
        mock=False,
        provider="deepseek",
        output_dir=str(tmp_path / "logs"),
        root_dir=ROOT,
    )
    assert result.get("ok") is False
    assert result.get("need_api_key") is True
    assert "api_key" in result.get("hint", "")


def test_dsh_credential_autoaligns_and_is_not_blocked(tmp_path, monkeypatch):
    """With DSH's own credential loaded, a real run is aligned (MCP == agent) and
    proceeds without a manual api_key. We patch resolve_llm so it runs offline."""
    monkeypatch.setattr(
        "mcp_server.core.resolve_llm",
        lambda *a, **k: MockLLM(model="mock"),
    )
    result = run_roundtable_impl(
        "API 对齐测试",
        council="experts",
        rounds=1,
        mock=False,
        provider="deepseek",
        output_dir=str(tmp_path / "logs"),
        root_dir=ROOT,
    )
    assert result.get("need_api_key") is not True
    assert result.get("final_summary")


def test_real_run_with_caller_api_is_not_blocked(tmp_path, monkeypatch):
    """Providing the caller's api clears the guard (MCP uses the SAME api as the agent)."""
    monkeypatch.setattr(
        "mcp_server.core.resolve_llm",
        lambda *a, **k: MockLLM(model="mock"),
    )
    result = run_roundtable_impl(
        "API 对齐测试",
        council="experts",
        rounds=1,
        mock=False,
        provider="deepseek",
        api_key="test-key",
        base_url="https://example.com",
        output_dir=str(tmp_path / "logs"),
        root_dir=ROOT,
    )
    assert result.get("need_api_key") is not True
    assert result.get("final_summary")
