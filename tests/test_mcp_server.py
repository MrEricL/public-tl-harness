"""Tests for agentic_translation.mcp_server: a hand-rolled MCP stdio JSON-RPC
server (no `mcp` SDK dependency). Exercised through an in-process fake
client driving newline-delimited JSON-RPC over StringIO pipes, and directly
against MCPServer.handle_line for the handshake/dispatch semantics."""

from __future__ import annotations

import io
import json

from agentic_translation.mcp_server import MCPServer, PROTOCOL_VERSION, serve
from agentic_translation.tool_surface import ToolSurface


def make_task(tmp_path):
    task_dir = tmp_path / "task"
    ToolSurface.create_task(
        task_dir, task_id="t", work_id="w", chapter_id="c",
        segments=[
            {"segment_id": "s1", "source_text": "你好", "draft_text": "Helo.",
             "context_text": "", "issue_ids": ["typo"]},
        ],
    )
    return task_dir


def test_initialize_echoes_protocol_version_and_advertises_tools(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PROTOCOL_VERSION},
    }))
    assert response["id"] == 1
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert "tools" in response["result"]["capabilities"]


def test_notifications_initialized_gets_no_response(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    assert response is None


def test_tools_list_returns_json_schema_per_tool(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    tools = {t["name"]: t for t in response["result"]["tools"]}
    assert set(tools) == set(ToolSurface.TOOL_NAMES)
    assert tools["edit_segment"]["inputSchema"]["required"] == ["segment_id", "expected_sha256", "edits"]
    assert tools["list_segments"]["inputSchema"]["type"] == "object"
    for tool in tools.values():
        assert "description" in tool and tool["description"]


def test_tools_call_success_returns_text_content_not_error(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "list_segments", "arguments": {}},
    }))
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is True
    assert payload["data"]["segments"][0]["segment_id"] == "s1"


def test_tools_call_failure_sets_is_error_true(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "read_segment", "arguments": {"segment_id": "nope"}},
    }))
    result = response["result"]
    assert result["isError"] is True
    payload = json.loads(result["content"][0]["text"])
    assert payload["ok"] is False


def test_tools_call_unknown_tool_name_is_isError_not_protocol_error(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "delete_everything", "arguments": {}},
    }))
    assert "error" not in response
    assert response["result"]["isError"] is True


def test_ping_replies_with_empty_result(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 6, "method": "ping"}))
    assert response == {"jsonrpc": "2.0", "id": 6, "result": {}}


def test_unknown_method_is_a_json_rpc_error(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "nope"}))
    assert response["error"]["code"] == -32601


def test_malformed_json_line_is_a_parse_error(tmp_path):
    server = MCPServer(str(make_task(tmp_path)))
    response = server.handle_line("{not json")
    assert response["error"]["code"] == -32700


def test_full_stdio_pipe_handshake_list_and_call(tmp_path):
    """Drive the server the way a real stdio client would: newline-delimited
    JSON-RPC over pipes, via the serve() loop, with a fake in-process client."""

    task_dir = make_task(tmp_path)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "list_segments", "arguments": {}}},
    ]
    in_stream = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    out_stream = io.StringIO()

    serve(str(task_dir), in_stream, out_stream)

    lines = [json.loads(line) for line in out_stream.getvalue().splitlines() if line.strip()]
    # The notification produced no line, so exactly 3 responses for 3 requests with ids.
    assert [entry["id"] for entry in lines] == [1, 2, 3]
    assert lines[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert {t["name"] for t in lines[1]["result"]["tools"]} == set(ToolSurface.TOOL_NAMES)
    assert json.loads(lines[2]["result"]["content"][0]["text"])["ok"] is True
