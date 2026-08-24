"""Entry point for the agent_roundtable MCP server.

Usage (from any directory):
    python mcp_entry.py                                # stdio transport (default)
    python mcp_entry.py --transport sse

The repository root is added to sys.path based on this file location, so the
server works even when launched from another working directory (for example
by Claude Code, which does not set a child cwd for stdio MCP servers).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp_server.__main__ import main  # noqa: E402


if __name__ == "__main__":
    main()
