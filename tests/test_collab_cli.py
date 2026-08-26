"""T17 tests: the collab workflow runner/CLI (company mode entry) + mode threading.
"""

from __future__ import annotations

import json
from pathlib import Path

from collab.cli import main
from collab.runner import get_collab_status, run_collaboration
from collab.runstore import RunStore


def _write_tasks(tmp_path: Path) -> Path:
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([
        {"id": "t1", "persona_id": "computing", "input": "评估符号计算影响", "expected_output": "给方案"},
        {"id": "t2", "persona_id": "history", "input": "评计算工具演变", "expected_output": "给结论"},
    ], ensure_ascii=False), encoding="utf-8")
    return p


def test_cli_run_wait_persists(tmp_path: Path, capsys):
    tasks = _write_tasks(tmp_path)
    db = tmp_path / "runs.db"
    code = main(["run", str(tasks), "--mock", "--db", str(db)])
    assert code == 0
    store = RunStore(db)
    runs = store.list()
    assert runs, "a run should be persisted"
    run_id = runs[0]["run_id"]
    assert runs[0]["status"] == "done"
    out = capsys.readouterr().out
    assert "run_id:" in out
    assert "done" in out


def test_cli_status_and_list(tmp_path: Path, capsys):
    tasks = _write_tasks(tmp_path)
    db = tmp_path / "runs.db"
    main(["run", str(tasks), "--mock", "--db", str(db)])
    store = RunStore(db)
    run_id = store.list()[0]["run_id"]
    main(["status", run_id, "--db", str(db)])
    out = capsys.readouterr().out
    assert '"status": "done"' in out
    main(["list", "--db", str(db)])
    out2 = capsys.readouterr().out
    assert run_id in out2


def test_cli_status_unknown(tmp_path: Path, capsys):
    db = tmp_path / "runs.db"
    main(["status", "nope", "--db", str(db)])
    out = capsys.readouterr().out
    assert '"not_found"' in out


def test_runner_threads_mode_parallel(tmp_path: Path):
    store = RunStore(tmp_path / "runs.db")
    tasks = [
        {"id": "t1", "persona_id": "computing", "input": "调研A"},
        {"id": "t2", "persona_id": "history", "input": "调研B"},
    ]
    run_id = run_collaboration(tasks, mock=True, mode="parallel", run_store=store)
    for _ in range(50):
        status = get_collab_status(run_id, run_store=store)["status"]
        if status not in ("running",):
            break
        import time
        time.sleep(0.05)
    st = get_collab_status(run_id, run_store=store)
    assert st["status"] == "done"
    assert "模式: parallel（experimental）" in st["final_report"]
