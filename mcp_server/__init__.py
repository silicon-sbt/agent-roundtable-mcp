"""MCP server that wraps agent_roundtable as callable MCP tools.

The service exposes roundtable discovery, knowledge-base search (use during
research), and multi-agent roundtable discussion (use after research, with your
findings injected via the context parameter).
"""

from __future__ import annotations

__version__ = "0.1.0"
