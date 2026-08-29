"""FastMCP server exposing agent_roundtable as MCP tools."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from roundtable.loader import PROJECT_ROOT

from .core import (
    get_agent_impl,
    list_agents_impl,
    list_councils_impl,
    list_knowledge_corpora_impl,
    list_reports_impl,
    plan_council_impl,
    read_report_impl,
    run_roundtable_impl,
    search_knowledge_impl,
)
from .llm_client import provider_status


def _estimate_total_steps(root_dir: Path, council: str, rounds: int) -> int:
    """Rough progress total: per round moderator + agents + summary, plus final + save."""
    try:
        from roundtable.loader import load_council

        agent_count = max(len(load_council(council, root_dir).members), 1)
    except Exception:
        agent_count = 3
    return max(rounds * (2 + agent_count) + 2, 1)


def create_server(root_dir: Path | str | None = None) -> FastMCP:
    """Build and return the agent-roundtable FastMCP server."""
    root = Path(root_dir).resolve() if root_dir else PROJECT_ROOT
    mcp = FastMCP(
        "agent-roundtable",
        instructions=(
            "Agent roundtable discussion service. Use list_councils / list_agents to pick a panel, use search_knowledge to consult local expert corpora during research, and run_roundtable to hold a multi-agent discussion optionally seeded with your research findings via the context parameter. Reports are saved under logs/; use list_reports / read_report to read them."
        ),
    )

    @mcp.tool(
        title="List roundtable councils",
        description="List available roundtable configurations (councils) with their expert members.",
    )
    def list_councils() -> list[dict[str, Any]]:
        return list_councils_impl(root)

    @mcp.tool(
        title="List expert agents",
        description="List all configured expert personas (domain experts and persona-inspired agents).",
    )
    def list_agents() -> list[dict[str, Any]]:
        return list_agents_impl(root)

    @mcp.tool(
        title="Get agent profile",
        description="Return the full persona card of one expert agent (worldview, style, strengths, profile).",
    )
    def get_agent(agent_id: str) -> dict[str, Any]:
        return get_agent_impl(agent_id, root)

    @mcp.tool(
        title="List knowledge corpora",
        description="List local RAG knowledge corpora (experts/ and people/ scopes).",
    )
    def list_knowledge_corpora() -> list[dict[str, Any]]:
        return list_knowledge_corpora_impl(root)

    @mcp.tool(
        title="Search knowledge base",
        description="Retrieve relevant passages from a local expert/person corpus for a query. Use during research to consult the panel source material before discussing.",
    )
    def search_knowledge(
        corpus_id: str,
        query: str,
        top_k: int = 5,
        knowledge_scope: str = "experts",
    ) -> dict[str, Any]:
        return search_knowledge_impl(
            corpus_id,
            query,
            top_k=top_k,
            knowledge_scope=knowledge_scope,
            root_dir=root,
        )

    @mcp.tool(
        title="Plan council for a topic",
        description="Analyse the council lineup for a topic BEFORE running the roundtable: returns current members, their roles, and guidance on how to adjust existing experts or add new ones (via run_roundtable's add_personas / adjust_personas parameters) to fit the discussion.",
    )
    def plan_council(
        topic: str = "",
        council: str = "experts",
    ) -> dict[str, Any]:
        return plan_council_impl(topic=topic, council=council, root_dir=root)

    @mcp.tool(
        title="Run agent roundtable",
        description="Hold a multi-agent roundtable discussion on a topic. Call plan_council first to review the lineup; then optionally customise it here: add_personas appends new experts (JSON array, each with name + role and optional worldview/speaking_style/strengths/weaknesses/catchphrases/profile), adjust_personas patches existing ones (JSON array, each with id + fields to override). Expert agents speak in turn under a moderator; the service produces a structured final summary and a Markdown report in logs/. Pass your research findings in the context parameter so the panel explicitly discusses them. Use mock=true for an offline demo, or set provider (deepseek/openai/openrouter/gemini/custom) with keys from .env. Real (non-mock) runs must pass the caller's api_key/base_url so the MCP and the requesting agent use the SAME api; otherwise it returns need_api_key. mock=true for offline demo.",
    )
    async def run_roundtable(
        ctx: Context,
        topic: str,
        council: str = "experts",
        rounds: int = 2,
        mock: bool = False,
        provider: str = "auto",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        context: str = "",
        add_personas: str = "",
        adjust_personas: str = "",
        output_dir: str = "logs",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        total = _estimate_total_steps(root, council, rounds)
        events: list[dict[str, Any]] = []
        lock = threading.Lock()

        def on_progress(event: dict[str, Any]) -> None:
            with lock:
                events.append(event)

        def runner() -> dict[str, Any]:
            return run_roundtable_impl(
                topic,
                council=council,
                rounds=rounds,
                mock=mock,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                context=context,
                add_personas=add_personas,
                adjust_personas=adjust_personas,
                output_dir=output_dir,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                root_dir=root,
                on_progress=on_progress,
            )

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, runner)
        reported = 0
        try:
            while True:
                done = future.done()
                with lock:
                    snapshot = events[reported:]
                    reported = len(events)
                for event in snapshot:
                    if event.get("event") == "done":
                        await ctx.report_progress(
                            float(reported),
                            float(total),
                            message=str(event.get("label", "")),
                        )
                if done:
                    break
                await asyncio.sleep(0.05)
            result = future.result()
        finally:
            if not future.done():
                future.cancel()
        await ctx.report_progress(float(total), float(total), message="Roundtable finished")
        return result

    @mcp.tool(
        title="List saved reports",
        description="List saved roundtable Markdown reports (newest first) under output_dir.",
    )
    def list_reports(
        output_dir: str = "logs",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return list_reports_impl(output_dir=output_dir, limit=limit)

    @mcp.tool(
        title="Read report",
        description="Read the content of a saved roundtable report (path from list_reports).",
    )
    def read_report(
        path: str,
        max_chars: int = 0,
        output_dir: str = "logs",
    ) -> dict[str, Any]:
        return read_report_impl(path, max_chars=max_chars, output_dir=output_dir)

    @mcp.tool(
        title="Provider status",
        description="Show which LLM providers are configured (from .env) and usable for roundtables.",
    )
    def provider_info() -> list[dict[str, Any]]:
        return provider_status(root)

    return mcp


__all__ = ["create_server"]
