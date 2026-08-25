"""V2 collaboration engine: async runner entry (T7).

Provides the M1 entry points that M3 will expose as MCP tools:

    run_collaboration(tasks, *, provider, mock, root_dir) -> run_id
        starts the collaboration on a background thread and returns a run id
        immediately (the graph itself stays synchronous in run_collab_sync).
    get_collab_status(run_id) -> dict
        polls the run: running/done/failed plus result summary (incl.
        overspend accounting from T6).
    stop_collab(run_id, reason) -> dict
        soft-stop: marks the run stopped; the report then records the reason.

Overspend responsibility (T6 review): the entry records overspend_tokens on
the run, and the final report states that the overrun is a wave-boundary
execution cost (parallel tasks cannot be interrupted mid-flight) - it is
accounted, not silently absorbed, and future pricing can use the real cost.
See docs/AGENT_GUIDE.md section 3.2.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from .graph import run_collab_sync
from .models import Task

_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _task_from_spec(spec: dict[str, Any]) -> Task:
    """Build a Task from a caller-provided definition dict (validation)."""
    required = ("id", "persona_id", "input")
    missing = [key for key in required if not str(spec.get(key, "")).strip()]
    if missing:
        raise ValueError("task spec missing required keys: " + ", ".join(missing))
    return Task.from_dict(spec)


def run_collaboration(
    tasks: list[dict[str, Any]],
    *,
    provider: str = "auto",
    mock: bool = False,
    root_dir: Any = None,
) -> str:
    """Start a collaboration on a background thread and return its run id.

    The caller polls with get_collab_status(run_id). M1 storage is in-memory
    (single process); M3 will wrap this pair as MCP tools.
    """
    if not tasks:
        raise ValueError("tasks must not be empty")
    parsed = [_task_from_spec(spec) for spec in tasks]
    run_id = _new_run_id()
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "created_at": created,
        "finished_at": None,
        "stop_reason": None,
        "provider": provider,
        "mock": mock,
        "state": None,
        "error": None,
    }
    with _LOCK:
        _RUNS[run_id] = record

    def _worker() -> None:
        try:
            state = run_collab_sync(parsed, provider=provider, mock=mock, root_dir=root_dir)
            with _LOCK:
                current = _RUNS.get(run_id)
                if current is None:
                    return
                current["state"] = state
                # A soft-stop must survive the worker finishing (otherwise the
                # thread would overwrite stopped back to done).
                if current.get("stop_reason"):
                    current["status"] = "stopped"
                else:
                    current["status"] = "done"
                current["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        except Exception as exc:
            with _LOCK:
                current = _RUNS.get(run_id)
                if current is None:
                    return
                current["status"] = "failed"
                current["error"] = str(exc)
                current["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


def get_collab_status(run_id: str) -> dict[str, Any]:
    """Return the current status and (once done) the result summary."""
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            return {"run_id": run_id, "status": "not_found", "error": "unknown run id"}
        summary: dict[str, Any] = {
            "run_id": run_id,
            "status": record["status"],
            "created_at": record["created_at"],
            "finished_at": record["finished_at"],
            "stop_reason": record["stop_reason"],
            "error": record["error"],
        }
        state = record["state"]
        if state is not None:
            summary["task_count"] = len(state.get("tasks", []))
            summary["results"] = [
                {
                    "id": r.get("id"),
                    "status": r.get("status"),
                    "failure_type": r.get("failure_type", ""),
                }
                for r in state.get("results", [])
            ]
            summary["token_total"] = state.get("token_total", 0)
            summary["overspend_tokens"] = _overspend_of(state)
            summary["final_report"] = state.get("final_report", "")
        return summary


def _overspend_of(state: dict[str, Any]) -> int:
    """T6 overspend accounting: sum overspend_tokens recorded on stopped tasks."""
    total = 0
    for result in state.get("results", []):
        if result.get("failure_type") == "global_budget":
            total = max(total, int(result.get("overspend_tokens", 0)))
    return total


def stop_collab(run_id: str, reason: str = "user requested") -> dict[str, Any]:
    """Soft-stop: mark the run stopped with a reason.

    M1 semantics: LangGraph invoke on the worker thread cannot be force-killed;
    the stop is recorded and surfaced in the final status so the caller can
    treat the run as cancelled. Hard cancellation is a M2 concern.
    """
    with _LOCK:
        record = _RUNS.get(run_id)
        if record is None:
            return {"run_id": run_id, "ok": False, "error": "unknown run id"}
        if record["status"] in ("done", "failed"):
            return {"run_id": run_id, "ok": False, "error": "run already finished: " + record["status"]}
        record["stop_reason"] = reason
        record["status"] = "stopped"
        return {"run_id": run_id, "ok": True, "stop_reason": reason}


def list_collab_runs() -> list[dict[str, Any]]:
    """List in-memory runs (newest first) for diagnostics/tests."""
    with _LOCK:
        runs = [
            {"run_id": rid, "status": r["status"], "created_at": r["created_at"]}
            for rid, r in _RUNS.items()
        ]
    runs.sort(key=lambda item: item["created_at"], reverse=True)
    return runs


__all__ = [
    "get_collab_status",
    "list_collab_runs",
    "run_collaboration",
    "stop_collab",
]
