"""collab CLI: the company workflow entry (T17).

Entry B (模式 B 公司) is a *workflow runner / CLI* - NOT an MCP server, and not
the DSH workflow tool (a foreground subagent orchestrator without persistence or
budget). This CLI drives the collab engine:

    python -m collab run <tasks.json> [--provider auto] [--mock] [--mode wave] [--report] [--db PATH]
    python -m collab status <run_id> [--db PATH]
    python -m collab report <run_id> [--db PATH]
    python -m collab list [--limit N] [--db PATH]
    python -m collab stop <run_id> [--reason "..."] [--db PATH]

Runs are persisted in the RunStore (default repo/logs/collab_runs.db) so history
and report survive across invocations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .runner import get_collab_status, list_collab_runs, run_collaboration, stop_collab
from .runstore import RunStore


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


DEFAULT_DB_PATH = _repo_root() / "logs" / "collab_runs.db"


def _store(db: str = "") -> RunStore:
    return RunStore(db if db else DEFAULT_DB_PATH)


def _load_tasks(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks must be a JSON array (list of task specs)")
    return data


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> int:
    tasks = _load_tasks(args.tasks)
    store = _store(getattr(args, "db", "") or "")
    run_id = run_collaboration(
        tasks,
        provider=args.provider,
        mock=args.mock,
        mode=args.mode,
        root_dir=args.root_dir or None,
        run_store=store,
    )
    print("run_id: " + run_id)
    # One-shot CLI: a background-thread run is killed when the process exits, so
    # `run` always polls to a terminal status (or --timeout), then prints the
    # status and optionally the report. On timeout/not_found it marks the run
    # stopped rather than leaving an orphaned "running" record.
    status: dict[str, Any] = {}
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        status = get_collab_status(run_id, run_store=store)
        if status.get("status") != "running":
            break
        time.sleep(0.5)
    if status.get("status") == "not_found":
        print("error: run not found", file=sys.stderr)
        _print_json(status)
        return 1
    if status.get("status") == "running":
        stop_collab(run_id, reason="cli timeout")
        print("warning: run did not finish within --timeout; marked stopped", file=sys.stderr)
        _print_json(status)
        return 1
    _print_json(status)
    if args.report and status.get("final_report"):
        print("\n--- final_report ---\n" + str(status.get("final_report")))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _print_json(get_collab_status(args.run_id, run_store=_store(getattr(args, "db", "") or "")))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    st = get_collab_status(args.run_id, run_store=_store(getattr(args, "db", "") or ""))
    print(st.get("final_report") or st.get("error") or "(no report yet)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for r in list_collab_runs(run_store=_store(getattr(args, "db", "") or ""))[: args.limit]:
        print(f'{r["run_id"]}\t{r["status"]}\t{r["created_at"]}')
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    _print_json(stop_collab(args.run_id, reason=args.reason))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collab", description="collab workflow runner (company mode / 模式 B)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="submit a company collaboration from a tasks JSON file")
    p_run.add_argument("tasks", help="path to a JSON file containing the tasks array")
    p_run.add_argument("--provider", default="auto")
    p_run.add_argument("--mock", action="store_true")
    p_run.add_argument("--mode", default="wave", choices=["wave", "parallel"])
    p_run.add_argument("--root-dir", default="")
    p_run.add_argument("--report", action="store_true", help="also print the final report")
    p_run.add_argument("--timeout", type=float, default=120.0, help="seconds to wait when --wait/--report")
    p_run.add_argument("--db", default="", help="RunStore db path (default repo/logs/collab_runs.db)")
    p_run.set_defaults(fn=cmd_run)

    p_status = sub.add_parser("status", help="print status of a run")
    p_status.add_argument("run_id")
    p_status.add_argument("--db", default="")
    p_status.set_defaults(fn=cmd_status)

    p_report = sub.add_parser("report", help="print the final report of a run")
    p_report.add_argument("run_id")
    p_report.add_argument("--db", default="")
    p_report.set_defaults(fn=cmd_report)

    p_list = sub.add_parser("list", help="list recent runs")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--db", default="")
    p_list.set_defaults(fn=cmd_list)

    p_stop = sub.add_parser("stop", help="soft-stop a running run")
    p_stop.add_argument("run_id")
    p_stop.add_argument("--reason", default="user requested")
    p_stop.add_argument("--db", default="")
    p_stop.set_defaults(fn=cmd_stop)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
