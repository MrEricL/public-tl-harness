"""A minimal MCP (Model Context Protocol) stdio server exposing the harness
lab's :class:`~agentic_translation.tool_surface.ToolSurface` tools.

This intentionally does not depend on the ``mcp`` Python SDK (it is not
installed in this environment). It implements just enough of the JSON-RPC 2.0
stdio transport for a tool-calling client (Claude Code's ``--mcp-config``,
Codex's ``mcp_servers.*``, or the in-process fake client in
``tests/test_mcp_server.py``) to drive one task directory:

* ``initialize`` -> echoes back ``protocolVersion`` and advertises
  ``capabilities.tools``.
* ``notifications/initialized`` -> a notification (no ``id``); acknowledged by
  doing nothing and not replying (notifications never get a response).
* ``tools/list`` -> JSON Schemas generated from
  :data:`agentic_translation.tool_surface.TOOL_JSON_SCHEMAS`.
* ``tools/call`` -> dispatches to :class:`ToolSurface`, wrapping the result as
  ``content: [{"type": "text", "text": <json>}]`` and setting ``isError`` on
  failure (as MCP expects -- ``isError`` communicates a tool-level failure,
  distinct from a JSON-RPC protocol error).
* ``ping`` -> replies with an empty result.

Transport: newline-delimited JSON-RPC 2.0 objects on stdin/stdout, one object
per line (a strict subset of the framing MCP stdio clients use; it avoids the
``Content-Length`` header framing some MCP transports use, which no client in
this lab requires).

Run as: ``python -m agentic_translation.mcp_server --task DIR``
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

from .tool_surface import TOOL_DESCRIPTIONS, TOOL_JSON_SCHEMAS, ToolSurface

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "harness-lab-tools"
SERVER_VERSION = "0.1.0"


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "inputSchema": TOOL_JSON_SCHEMAS[name],
        }
        for name in ToolSurface.TOOL_NAMES
    ]


class MCPServer:
    """Handles one JSON-RPC request/notification at a time, synchronously."""

    def __init__(self, task_dir: str):
        self.surface = ToolSurface(task_dir)

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """Parse and dispatch one line. Returns a response object, or None for
        a notification (which never gets a JSON-RPC response) or a parse error
        we cannot even attribute an id to.
        """

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return _error(None, -32700, "Parse error: invalid JSON")
        if not isinstance(message, dict):
            return _error(None, -32600, "Invalid request: expected a JSON object")

        method = message.get("method")
        request_id = message.get("id")
        is_notification = "id" not in message

        if not isinstance(method, str):
            if is_notification:
                return None
            return _error(request_id, -32600, "Invalid request: missing 'method'")

        if method == "notifications/initialized":
            return None  # No response for notifications, per JSON-RPC 2.0.
        if method.startswith("notifications/"):
            return None

        try:
            result = self._dispatch(method, message.get("params") or {})
        except _MethodNotFound as exc:
            if is_notification:
                return None
            return _error(request_id, -32601, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            if is_notification:
                return None
            return _error(request_id, -32603, f"Internal error: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "capabilities": {"tools": {"listChanged": False}},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": _tool_definitions()}
        if method == "tools/call":
            return self._tools_call(params)
        raise _MethodNotFound(f"Unknown method '{method}'")

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in ToolSurface.TOOL_NAMES:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "ok": False,
                    "message": f"Unknown tool '{name}'. Known tools: {', '.join(ToolSurface.TOOL_NAMES)}.",
                })}],
                "isError": True,
            }
        observation = self.surface.call(str(name), arguments if isinstance(arguments, dict) else {})
        return {
            "content": [{"type": "text", "text": json.dumps(observation, ensure_ascii=False)}],
            "isError": not bool(observation.get("ok")),
        }


class _MethodNotFound(Exception):
    pass


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(task_dir: str, in_stream: TextIO, out_stream: TextIO) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin closes."""

    server = MCPServer(task_dir)
    for line in in_stream:
        line = line.strip()
        if not line:
            continue
        response = server.handle_line(line)
        if response is not None:
            out_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            out_stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentic_translation.mcp_server")
    parser.add_argument("--task", required=True, help="Task directory (see tool_surface.ToolSurface).")
    args = parser.parse_args(argv)
    serve(args.task, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
