"""Plain, dependency-free tool implementations for the MCP server.

The FastMCP layer in server.py is a thin wrapper around these functions so
they can be unit-tested and reused without an MCP session.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rag.config import discover_corpora
from rag.retriever import format_retrieved_context, get_retriever, source_files
from roundtable.discovery import discover_configured_agent_ids
from roundtable.graph import build_roundtable_graph, run_roundtable
from roundtable.loader import PROJECT_ROOT, load_council, load_council_personas, load_persona
from roundtable.logger import save_markdown_log
from roundtable.state import Persona

from .llm_client import load_agent_credential, resolve_llm

ProgressCallback = Callable[[dict], None]

DEFAULT_OUTPUT_DIR = "logs"


def _root(root_dir: Path | str | None) -> Path:
    return Path(root_dir).resolve() if root_dir else PROJECT_ROOT


def _resolve_output_dir(output_dir: str, root: Path) -> Path:
    """Resolve output_dir: absolute stays absolute; relative is rooted at repo root."""
    path = Path(output_dir)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_councils_impl(root_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """List available roundtable councils with their members."""
    root = _root(root_dir)
    councils_dir = root / "config" / "councils"
    councils: list[dict[str, Any]] = []
    if councils_dir.exists():
        for path in sorted(councils_dir.glob("*.yaml")):
            try:
                council = load_council(path.stem, root)
            except Exception as exc:  # keep one bad file from breaking discovery
                councils.append({"name": path.stem, "error": str(exc)})
                continue
            councils.append(
                {
                    "name": council.name,
                    "description": council.description,
                    "members": council.members,
                    "moderator": council.moderator,
                }
            )
    return councils


def list_agents_impl(root_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """List all configured expert personas (domain experts + persona inspired)."""
    root = _root(root_dir)
    agents: list[dict[str, Any]] = []
    for agent_id in discover_configured_agent_ids(root):
        try:
            persona = load_persona(agent_id, root)
        except Exception as exc:
            agents.append({"id": agent_id, "error": str(exc)})
            continue
        agents.append(
            {
                "id": persona.id,
                "name": persona.name,
                "role": persona.role,
                "agent_type": persona.agent_type,
                "rag_expert_name": persona.rag_expert_name,
                "knowledge_scope": persona.knowledge_scope,
            }
        )
    return agents


def get_agent_impl(agent_id: str, root_dir: Path | str | None = None) -> dict[str, Any]:
    """Return the full persona card of one expert agent."""
    root = _root(root_dir)
    if not agent_id.strip():
        raise ValueError("agent_id must not be empty")
    persona = load_persona(agent_id.strip(), root)
    return persona.to_dict()


def list_knowledge_corpora_impl(
    root_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List local RAG knowledge corpora (experts/ and people/ scopes)."""
    root = _root(root_dir)
    return [
        {"corpus_id": corpus.corpus_id, "knowledge_scope": corpus.knowledge_scope}
        for corpus in discover_corpora(root)
    ]


def search_knowledge_impl(
    corpus_id: str,
    query: str,
    *,
    top_k: int = 5,
    knowledge_scope: str = "experts",
    root_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Retrieve relevant passages from a local corpus for a query."""
    root = _root(root_dir)
    if not corpus_id.strip():
        raise ValueError("corpus_id must not be empty")
    if not query.strip():
        raise ValueError("query must not be empty")
    if knowledge_scope not in ("experts", "people"):
        raise ValueError("knowledge_scope must be experts or people")
    retriever = get_retriever(
        corpus_id.strip(),
        top_k=max(1, min(int(top_k), 20)),
        knowledge_scope=knowledge_scope,  # type: ignore[arg-type]
        root_dir=root,
    )
    chunks = retriever.invoke(query.strip())
    return {
        "corpus_id": corpus_id.strip(),
        "knowledge_scope": knowledge_scope,
        "query": query.strip(),
        "count": len(chunks),
        "context": format_retrieved_context(chunks),
        "sources": source_files(chunks),
    }


# ---------------------------------------------------------------------------
# Roundtable
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Turn an arbitrary name into a usable persona id.

    Keeps unicode word characters (including CJK) so Chinese persona names
    produce readable ids instead of colliding on the fallback.
    """
    import re

    slug = re.sub(r"[^\w-]+", "_", name.strip().lower()).strip("_")
    return slug or "custom_agent"


def _parse_persona_list(raw: str, field: str) -> list[dict[str, Any]]:
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(field + " must be a JSON array of persona objects: " + str(exc)) from exc
    if not isinstance(data, list):
        raise ValueError(field + " must be a JSON array")
    return data


def apply_persona_changes(
    personas: list[Persona],
    add_personas: str = "",
    adjust_personas: str = "",
) -> list[Persona]:
    """Adjust existing personas by id, then append new personas from JSON specs."""
    result = list(personas)
    for override in _parse_persona_list(adjust_personas, "adjust_personas"):
        if not isinstance(override, dict) or not override.get("id"):
            raise ValueError("adjust_personas entries need an id")
        pid = str(override["id"])
        idx = next((i for i, p in enumerate(result) if p.id == pid), None)
        if idx is None:
            raise ValueError(
                "adjust_personas: unknown persona id '" + pid + "'. "
                "Available: " + ", ".join(p.id for p in personas),
            )
        fields = {k: v for k, v in override.items() if k != "id" and hasattr(Persona, k)}
        result[idx] = replace(result[idx], **fields)
    for spec in _parse_persona_list(add_personas, "add_personas"):
        if not isinstance(spec, dict):
            raise ValueError("add_personas entries must be objects")
        name = str(spec.get("name", "")).strip()
        role = str(spec.get("role", "")).strip()
        if not name or not role:
            raise ValueError("add_personas entries need at least name and role")
        pid = str(spec.get("id") or _slugify(name))
        if pid in {p.id for p in result}:
            raise ValueError("add_personas: id '" + pid + "' already exists")
        result.append(
            Persona(
                id=pid,
                name=name,
                role=role,
                worldview=str(spec.get("worldview", "")),
                speaking_style=str(spec.get("speaking_style", "")),
                strengths=([str(item) for item in spec["strengths"]] if isinstance(spec.get("strengths"), list) else []),
                weaknesses=([str(item) for item in spec["weaknesses"]] if isinstance(spec.get("weaknesses"), list) else []),
                catchphrases=([str(item) for item in spec["catchphrases"]] if isinstance(spec.get("catchphrases"), list) else []),
                llm_config=dict(spec.get("llm_config", {})),
                rag_expert_name=str(spec["rag_expert_name"]) if spec.get("rag_expert_name") else None,
                knowledge_scope=str(spec["knowledge_scope"]) if spec.get("knowledge_scope") else None,
                agent_type=str(spec.get("agent_type") or "domain_expert"),
                profile=(dict(spec["profile"]) if isinstance(spec.get("profile"), dict) else None),
            )
        )
    return result


def _run_roundtable_with_personas(
    *,
    topic: str,
    council: Any,
    personas: list[Persona],
    rounds: int,
    llm: Any,
    root: Path,
    output_dir: Path,
    initial_messages: list[dict[str, Any]],
    on_progress: ProgressCallback | None,
) -> dict[str, Any]:
    """Run the LangGraph roundtable with a custom persona lineup."""
    app = build_roundtable_graph(
        council,
        personas,
        llm,
        root_dir=root,
        agent_llms={persona.id: llm for persona in personas},
        progress_callback=on_progress,
    )
    start_round = max(
        [int(m.get("round", 0)) for m in initial_messages if isinstance(m.get("round", 0), int)]
        or [0],
    )
    initial_state: dict[str, Any] = {
        "topic": topic,
        "round": start_round,
        "max_rounds": start_round + rounds,
        "council_name": council.name,
        "council_description": council.description,
        "current_speaker": "",
        "personas": [persona.to_dict() for persona in personas],
        "messages": list(initial_messages),
        "round_summaries": [],
        "final_summary": "",
    }
    if on_progress is not None:
        on_progress({"event": "start", "stage": "save_log", "label": "Saving Markdown report"})
    result = app.invoke(initial_state)
    log_path = save_markdown_log(result, output_dir=output_dir)
    if on_progress is not None:
        on_progress(
            {
                "event": "done",
                "stage": "save_log",
                "label": "Saved Markdown report",
                "log_path": log_path,
            }
        )
    result["log_path"] = log_path
    return result


def plan_council_impl(
    topic: str = "",
    council: str = "experts",
    root_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Analyse the council lineup for a topic before the roundtable starts.

    Helps the calling agent decide whether to adjust existing experts or add
    new ones via run_roundtable's add_personas / adjust_personas parameters.
    """
    root = _root(root_dir)
    council_obj, personas = load_council_personas(council, root)
    members = [
        {
            "id": persona.id,
            "name": persona.name,
            "role": persona.role,
            "agent_type": persona.agent_type,
            "rag_expert_name": persona.rag_expert_name,
        }
        for persona in personas
    ]
    lineup = "、".join(persona.name + "（" + persona.role + "）" for persona in personas)
    suggestions = [
        "当前阵容：" + lineup,
        "如需补充视角：调用 run_roundtable 时传 add_personas（JSON 数组，每项至少含 name 和 role，可带 worldview / speaking_style / strengths / weaknesses / catchphrases / rag_expert_name / profile）。",
        "如需调整现有专家：传 adjust_personas（JSON 数组，每项含 id 与要覆盖的字段，如 role / worldview / speaking_style）。",
        "可用 list_agents 查看已有专家模板，用 search_knowledge 查本地语料判断视角缺口。",
        "提示：若你给 council 新增/调整了专家，请记得为这些专家补充语料（person_crawl_agent / knowledge/），否则其知识库可能为空、讨论会缺依据。",
    ]
    return {
        "topic": topic,
        "council": council_obj.name,
        "description": council_obj.description,
        "member_count": len(members),
        "members": members,
        "suggestions": suggestions,
    }


def _serialize_result(state: Any) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in state.get("messages", []):
        messages.append(
            {
                "round": int(message.get("round", 0)),
                "speaker": str(message.get("speaker", "")),
                "speaker_id": str(message.get("speaker_id", "")),
                "role": str(message.get("role", "")),
                "type": str(message.get("type", "")),
                "content": str(message.get("content", "")),
                "references": [str(ref) for ref in message.get("references", [])],
                "epistemic_tags": [str(tag) for tag in message.get("epistemic_tags", [])],
                "llm_provider": str(message.get("llm_provider", "")),
                "llm_model": str(message.get("llm_model", "")),
            }
        )
    return {
        "topic": str(state.get("topic", "")),
        "council_name": str(state.get("council_name", "")),
        "council_description": str(state.get("council_description", "")),
        "rounds": int(state.get("round", 0)),
        "final_summary": str(state.get("final_summary", "")),
        "round_summaries": [str(item) for item in state.get("round_summaries", [])],
        "log_path": str(state.get("log_path", "")),
        "message_count": len(messages),
        "messages": messages,
    }


def run_roundtable_impl(
    topic: str,
    *,
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
    output_dir: str = DEFAULT_OUTPUT_DIR,
    temperature: float = 0.7,
    max_output_tokens: int = 4096,
    root_dir: Path | str | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run a full roundtable discussion and return transcript + final summary."""
    root = _root(root_dir)
    if not topic.strip():
        raise ValueError("topic must not be empty")
    rounds = int(rounds)
    if rounds < 1 or rounds > 10:
        raise ValueError("rounds must be between 1 and 10")

    # API-alignment guard: guarantee the MCP and the requesting agent use the SAME
    # api. A real (non-mock) roundtable must be given the caller's api_key/base_url
    # so it never silently falls back to a .env key that may differ from the agent.
    dsh_aligned = (not mock) and load_agent_credential(root)
    if not mock and not (api_key or base_url) and not dsh_aligned:
        return {
            "ok": False,
            "need_api_key": True,
            "provider": provider or "auto",
            "message": (
                "为保证 MCP 与请求方 agent 使用同一 API，请传入 provider + api_key（或 base_url）。"
                "当前未提供，MCP 不会静默改用 .env 中可能不一致的 key。"
            ),
            "hint": "从你（agent）正在使用的模型配置取得 provider、api_key、base_url 后传入；要离线演示可用 mock=true。",
        }

    llm = resolve_llm(
        provider if not mock else "mock",
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        root_dir=root,
    )

    initial_messages: list[dict[str, Any]] = []
    if context and context.strip():
        initial_messages.append(
            {
                "round": 0,
                "speaker": "调研背景",
                "speaker_id": "research",
                "role": "调研员提供的背景资料",
                "type": "context",
                "content": context.strip(),
                "references": [],
                "reference_details": [],
                "epistemic_tags": ["调研背景"],
                "llm_provider": "research-notes",
                "llm_model": "user-provided",
            }
        )

    if add_personas or adjust_personas:
        council_obj, personas = load_council_personas(council, root)
        personas = apply_persona_changes(personas, add_personas, adjust_personas)
        result = _run_roundtable_with_personas(
            topic=topic,
            council=council_obj,
            personas=personas,
            rounds=rounds,
            llm=llm,
            root=root,
            output_dir=_resolve_output_dir(output_dir, root),
            initial_messages=initial_messages,
            on_progress=on_progress,
        )
    else:
        result = run_roundtable(
            topic=topic,
            council_name=council,
            rounds=rounds,
            llm=llm,
            root_dir=root,
            output_dir=_resolve_output_dir(output_dir, root),
            initial_messages=initial_messages,
            progress_callback=on_progress,
        )
    serialized = _serialize_result(result)
    if add_personas or adjust_personas:
        serialized["reminder"] = (
            "提示：本次新增/调整了专家。请为这些专家补充语料（如调用 person_crawl_agent 抓取或向 knowledge/ 添加资料），"
            "否则其知识库可能为空、讨论会缺依据。补充什么内容由你（调用方）判断。"
        )
    return serialized


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def list_reports_impl(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List saved roundtable Markdown reports, newest first."""
    directory = _resolve_output_dir(output_dir, PROJECT_ROOT)
    if not directory.exists():
        return []
    files = sorted(
        (path for path in directory.glob("*.md") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    reports: list[dict[str, Any]] = []
    for path in files[: max(1, min(int(limit), 200))]:
        stat = path.stat()
        reports.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                    timespec="seconds"
                ),
            }
        )
    return reports


def read_report_impl(
    path: str,
    max_chars: int = 0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Read the content of a saved roundtable report.

    Only files under ``output_dir`` are readable; absolute paths or ``..``
    escapes outside it are rejected so the MCP client cannot read arbitrary
    files on the host.
    """
    report_path = Path(path)
    base = _resolve_output_dir(output_dir, PROJECT_ROOT)
    if not report_path.is_absolute():
        report_path = base / report_path
    resolved = report_path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("Report path must be inside output_dir: " + path)
    if not resolved.exists():
        raise FileNotFoundError("Report not found: " + path)
    text = resolved.read_text(encoding="utf-8")
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated; full size %d chars]" % len(text)
        truncated = True
    return {
        "path": str(report_path.resolve()),
        "name": report_path.name,
        "chars": len(text),
        "truncated": truncated,
        "content": text,
    }


__all__ = [
    "list_councils_impl",
    "list_agents_impl",
    "get_agent_impl",
    "list_knowledge_corpora_impl",
    "search_knowledge_impl",
    "plan_council_impl",
    "run_roundtable_impl",
    "list_reports_impl",
    "read_report_impl",
]
