"""Project adapter for the shared local Claude/Codex CLI transport."""

from __future__ import annotations

from typing import Optional

from llm_gateway.local_cli import run_local_cli

from .environment import PROJECT_ROOT, merged_environment


def run_cli(
    backend: str,
    text: str,
    model: str = "",
    effort: str = "",
    *,
    env_override: Optional[dict[str, str]] = None,
) -> str:
    return run_local_cli(
        backend,
        text,
        model,
        effort,
        project_root=PROJECT_ROOT,
        env_override=merged_environment(env_override),
    )
