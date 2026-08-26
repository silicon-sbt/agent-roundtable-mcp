"""Tests for the MCP server tool implementations (no MCP session needed)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_server.core import (
    plan_council_impl,
    get_agent_impl,
    list_agents_impl,
    list_councils_impl,
    list_knowledge_corpora_impl,
    list_reports_impl,
    read_report_impl,
    run_roundtable_impl,
    search_knowledge_impl,
)
from mcp_server.llm_client import resolve_llm
from mcp_server.server import create_server

ROOT = Path(__file__).resolve().parent.parent


def test_server_registers_expected_tools():
    async def _check():
        server = create_server(ROOT)
        names = {tool.name for tool in await server.list_tools()}
        assert {
            "list_councils",
            "list_agents",
            "get_agent",
            "search_knowledge",
            "run_roundtable",
            "list_reports",
            "read_report",
        } <= names

    asyncio.run(_check())


def test_list_councils_contains_experts():
    councils = list_councils_impl(ROOT)
    names = {council["name"] for council in councils}
    assert "experts" in names
    experts = next(council for council in councils if council["name"] == "experts")
    assert "macroeconomics" in experts["members"]


def test_list_agents_contains_domain_experts():
    agents = list_agents_impl(ROOT)
    ids = {agent["id"] for agent in agents}
    assert {
        "macroeconomics",
        "investing",
        "computing",
        "philosophy",
        "history",
    } <= ids


def test_get_agent_profile():
    profile = get_agent_impl("macroeconomics", ROOT)
    assert profile["role"]
    assert profile["agent_type"] == "domain_expert"


def test_list_knowledge_corpora_is_safe():
    # The repo may have only .gitkeep placeholders; this must not raise.
    assert isinstance(list_knowledge_corpora_impl(ROOT), list)


def test_search_knowledge_empty_corpus_is_safe():
    result = search_knowledge_impl("macroeconomics", "总需求与货币政策", root_dir=ROOT)
    assert result["corpus_id"] == "macroeconomics"
    assert result["count"] >= 0


def test_resolve_llm_mock():
    llm = resolve_llm("mock", root_dir=ROOT)
    assert llm.provider_name == "mock"
    assert llm.generate("话题：测试")


def test_run_roundtable_mock(tmp_path):
    output_dir = tmp_path / "logs"
    result = run_roundtable_impl(
        "MCP 服务如何帮助 agent 做调研与讨论",
        council="experts",
        rounds=1,
        mock=True,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    assert result["topic"]
    assert result["final_summary"]
    assert result["message_count"] >= 7, result["message_count"]
    log_path = Path(result["log_path"])
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "Final Summary" in text


def test_context_seed_appears_in_transcript(tmp_path):
    output_dir = tmp_path / "logs"
    result = run_roundtable_impl(
        "测试调研背景注入",
        council="china_debt",
        rounds=1,
        mock=True,
        context="调研发现：本地语料与在线资料存在分歧。",
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    assert any(message["speaker_id"] == "research" for message in result["messages"])


def test_list_and_read_reports(tmp_path):
    output_dir = tmp_path / "logs"
    run_roundtable_impl(
        "报告读写测试",
        council="experts",
        rounds=1,
        mock=True,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    reports = list_reports_impl(str(output_dir))
    assert reports
    content = read_report_impl(reports[0]["path"], output_dir=str(output_dir))
    assert content["content"]
    assert not content["truncated"]


def test_run_roundtable_requires_topic():
    with pytest.raises(ValueError):
        run_roundtable_impl("   ", council="experts", rounds=1, mock=True, root_dir=ROOT)

def test_tool_schemas_avoid_dsh_unsupported_vocabulary():
    """Regression: dsh-mcp-client rejects anyOf and type arrays via
    assertSupportedJsonSchema; optional params must stay non-Optional.
    """
    async def _check():
        server = create_server(ROOT)
        tools = await server.list_tools()
        assert tools, "expected at least one tool"
        for tool in tools:
            schema = tool.inputSchema
            assert "anyOf" not in schema, tool.name
            assert "oneOf" not in schema, tool.name
            for prop in schema.get("properties", {}).values():
                assert isinstance(prop.get("type"), str), (
                    tool.name, prop, "type must be a single string",
                )

    asyncio.run(_check())


def test_read_report_rejects_paths_outside_output_dir(tmp_path):
    output_dir = tmp_path / "logs"
    run_roundtable_impl(
        "路径安全测试",
        council="experts",
        rounds=1,
        mock=True,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError):
        read_report_impl(str(outside), output_dir=str(output_dir))
    # absolute path inside the dir still works
    reports = list_reports_impl(str(output_dir))
    assert reports
    content = read_report_impl(reports[0]["path"], output_dir=str(output_dir))
    assert content["content"]

def test_plan_council_reports_lineup(tmp_path):
    result = plan_council_impl("AI 对就业的影响", council="experts", root_dir=ROOT)
    assert result["council"] == "experts"
    assert result["member_count"] >= 5
    ids = {m["id"] for m in result["members"]}
    assert "macroeconomics" in ids
    assert result["suggestions"], "expected lineup guidance"


def test_run_roundtable_with_added_persona(tmp_path):
    output_dir = tmp_path / "logs"
    add = json.dumps([
        {
            "name": "Energy Expert",
            "role": "能源与产业政策专家",
            "worldview": "从能源供需、地缘政治与技术替代分析问题",
            "speaking_style": "先讲约束再讲判断",
            "strengths": ["能源供需分析"],
            "weaknesses": ["可能低估金融波动"],
        }
    ])
    result = run_roundtable_impl(
        "能源转型对经济的影响",
        council="experts",
        rounds=1,
        mock=True,
        add_personas=add,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    speakers = {m["speaker_id"] for m in result["messages"]}
    assert "energy_expert" in speakers, speakers


def test_run_roundtable_adjust_persona(tmp_path):
    output_dir = tmp_path / "logs"
    adjust = json.dumps([
        {"id": "history", "role": "技术史与信息史专家", "speaking_style": "简洁"}
    ])
    result = run_roundtable_impl(
        "技术史视角测试",
        council="experts",
        rounds=1,
        mock=True,
        adjust_personas=adjust,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    history_msg = next(m for m in result["messages"] if m["speaker_id"] == "history")
    assert history_msg["content"]


def test_run_roundtable_add_persona_bad_json(tmp_path):
    with pytest.raises(ValueError):
        run_roundtable_impl(
            "测试",
            council="experts",
            rounds=1,
            mock=True,
            add_personas="not json",
            root_dir=ROOT,
        )


def test_run_roundtable_adjust_unknown_persona(tmp_path):
    with pytest.raises(ValueError, match="unknown persona"):
        run_roundtable_impl(
            "测试",
            council="experts",
            rounds=1,
            mock=True,
            adjust_personas=json.dumps([{"id": "nobody", "role": "x"}]),
            root_dir=ROOT,
        )

def test_add_personas_chinese_names_no_collision(tmp_path):
    """Chinese persona names must produce distinct ids (regression)."""
    output_dir = tmp_path / "logs"
    add = json.dumps([
        {"name": "训诂学专家", "role": "训诂与文字学专家"},
        {"name": "技术翻译史专家", "role": "近代科技翻译史专家"},
    ], ensure_ascii=False)
    result = run_roundtable_impl(
        "中文专家名测试",
        council="experts",
        rounds=1,
        mock=True,
        add_personas=add,
        output_dir=str(output_dir),
        root_dir=ROOT,
    )
    speakers = {m["speaker_id"] for m in result["messages"]}
    assert "训诂学专家" in speakers, speakers
    assert "技术翻译史专家" in speakers, speakers


def test_plan_council_reminds_to_supplement_corpus():
    result = plan_council_impl("x", council="experts", root_dir=ROOT)
    assert any("补充语料" in s for s in result["suggestions"])


def test_run_roundtable_add_personas_returns_reminder(tmp_path):
    output_dir = tmp_path / "logs"
    add = json.dumps([{"name": "量化分析师", "role": "量化投资"}])
    result = run_roundtable_impl(
        "测试新增专家", council="experts", rounds=1, mock=True,
        add_personas=add, output_dir=str(output_dir), root_dir=ROOT,
    )
    assert "reminder" in result
    assert "补充语料" in result["reminder"]
