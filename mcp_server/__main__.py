"""CLI entry point for the agent_roundtable MCP server.

Usage:
    python -m mcp_server                       # stdio transport (default)
    python -m mcp_server --transport sse
    python -m mcp_server --transport streamable-http
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the server runnable from any working directory: add the repository root
# (parent of this package) to sys.path when it is not already importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .server import create_server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="agent_roundtable MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio, for DSH / Claude Desktop / Cursor)",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="agent_roundtable repository root (default: this repo)",
    )
    args = parser.parse_args()

    server = create_server(root_dir=args.root_dir)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
