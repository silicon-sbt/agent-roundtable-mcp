"""V2 collaboration engine (mode B: weak-decentralised execution).

M1 scope: task model + state machine + L2 audit + wave executor +
horizontal collaboration + arbitration + recovery/budget + async runner.
See docs/AGENT_GUIDE.md.
"""

from __future__ import annotations

from .models import CollabMessage, Task, TaskAudit, TaskStatus
from .runner import get_collab_status, run_collaboration, stop_collab
from .state_machine import TaskStateMachine, can_transition

__all__ = [
    "Task",
    "TaskStatus",
    "TaskAudit",
    "CollabMessage",
    "TaskStateMachine",
    "can_transition",
    "run_collaboration",
    "get_collab_status",
    "stop_collab",
]
