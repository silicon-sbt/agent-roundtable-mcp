"""Project adapter for the shared CLIProxyAPI HTTP transport."""

from __future__ import annotations

from typing import Optional

from llm_gateway.proxy import (
    antigravity_model_variant,
    resolve_model_id,
    run_proxy,
)

from .environment import PROJECT_ROOT, merged_environment


def run_cli(
    backend: str,
    text: str,
    model: str = "",
    effort: str = "",
    *,
    env_override: Optional[dict[str, str]] = None,
) -> str:
    return run_proxy(
        backend,
        text,
        model,
        effort,
        project_root=PROJECT_ROOT,
        env_override=merged_environment(env_override),
    )
