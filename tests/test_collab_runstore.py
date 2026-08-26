"""T16 tests: RunStore persistence (cross-instance), run_collaboration persisting
its summary, historical get_collab_status, and list merging live + history.
"""

from __future__ import annotations

import time
from pathlib import Path

from collab.runstore import RunStore
from collab.runner import get_collab_status, list_collab_runs, run_collaboration


def _task_spec(task_id: str, persona: str, input_text: str) -> dict:
    return {"id": task_id, "persona_id": persona, "input": input_text}


def test_runstore_roundtrip_and_cross_instance(tmp_path: Path):
    db = tmp_path / "runs.db"
    store1 = RunStore(db)
    store1.save({
        "run_id": "r-1", "status": "done", "created_at": "2026-08-25T00:00:00+00:00",
        "finished_at": "2026-08-25T00:00:10+00:00", "stop_reason": None,
        "provider": "mock", "mock": True,
        "summary": {"run_id": "r-1", "status": "done", "task_count": 2, "final_report": "# report"},
    })
    # A fresh store on the same path sees the record (restart survives).
    store2 = RunStore(db)
    got = store2.get("r-1")
    assert got is not None
    assert got["status"] == "done"
    assert got["summary"]["task_count"] == 2
    assert got["summary"]["final_report"] == "# report"


def test_run_collaboration_persists_summary(tmp_path: Path):
    store = RunStore(tmp_path / "runs.db")
    run_id = run_collaboration(
        [_task_spec("t1", "computing", "调研A"), _task_spec("t2", "history", "调研B")],
        mock=True,
        run_store=store,
    )
    # Poll until the worker finishes.
    for _ in range(50):
        status = get_collab_status(run_id, run_store=store)["status"]
        if status not in ("running",):
            break
        time.sleep(0.05)
    record = store.get(run_id)
    assert record is not None
    assert record["status"] == "done"
    assert "final_report" in record["summary"]


def test_get_collab_status_reads_persisted_history(tmp_path: Path):
    store = RunStore(tmp_path / "runs.db")
    store.save({
        "run_id": "r-hist", "status": "done", "created_at": "2026-08-25T00:00:00+00:00",
        "finished_at": "2026-08-25T00:00:10+00:00", "stop_reason": None,
        "provider": "mock", "mock": True,
        "summary": {"run_id": "r-hist", "status": "done", "task_count": 1, "final_report": "# hist"},
    })
    # run_id not in this process memory -> falls back to the persisted store.
    status = get_collab_status("r-hist", run_store=store)
    assert status["status"] == "done"
    assert status["final_report"] == "# hist"


def test_list_collab_runs_merges_live_and_history(tmp_path: Path):
    store = RunStore(tmp_path / "runs.db")
    # A historical (restarted) run that is NOT in this process's memory.
    store.save({
        "run_id": "r-old", "status": "done", "created_at": "2026-08-24T00:00:00+00:00",
        "finished_at": None, "stop_reason": None, "provider": "mock", "mock": True,
        "summary": {"run_id": "r-old", "status": "done"},
    })
    # A live run added now.
    run_id = run_collaboration([_task_spec("t1", "computing", "调研A")], mock=True, run_store=store)
    ids = [r["run_id"] for r in list_collab_runs(run_store=store)]
    assert run_id in ids
    assert "r-old" in ids
