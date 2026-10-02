"""A bash-callable CLI over the harness lab's tool surface.

    python -m agentic_translation.tools_cli --task DIR <tool> [json-args]

``json-args`` is one JSON object given as a single argument, e.g.:

    python -m agentic_translation.tools_cli --task runs/lab-x/t1 read_segment '{"segment_id":"s0001"}'

Prints exactly one JSON object (the tool observation) to stdout and exits 0
whether or not the tool call itself succeeded -- ``{"ok": false, ...}`` is a
normal, well-formed observation, not a CLI failure. The process exits nonzero
only for a usage error (bad arguments, malformed JSON, unknown tool, missing
task directory), whose message is printed to stderr.

This is the surface a shell-only harness (e.g. Pi, or any agent that can only
run bash commands) can drive directly, without an MCP client.
"""

from __future__ import annotations

import argparse
import json
import sys

from .tool_surface import ToolSurface, ToolSurfaceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentic_translation.tools_cli")
    parser.add_argument("--task", required=True, help="Task directory created by ToolSurface.create_task.")
    parser.add_argument("tool", choices=list(ToolSurface.TOOL_NAMES), help="Tool name.")
    parser.add_argument(
        "json_args",
        nargs="?",
        default="{}",
        help="A single JSON object of tool arguments (default: '{}').",
    )
    args = parser.parse_args(argv)

    try:
        tool_args = json.loads(args.json_args)
    except json.JSONDecodeError as exc:
        print(f"usage error: json-args is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(tool_args, dict):
        print("usage error: json-args must be a JSON object", file=sys.stderr)
        return 2

    try:
        surface = ToolSurface(args.task)
        observation = surface.call(args.tool, tool_args)
    except ToolSurfaceError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(observation, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
