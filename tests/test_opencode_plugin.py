"""Small executable probes for the OpenCode plugin's built-in task fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_opencode_sees_only_one_plugin_export() -> None:
    """OpenCode invokes every module export as a plugin factory."""
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is not installed")
    plugin = (
        Path(__file__).resolve().parents[1]
        / "minireplay"
        / "assets"
        / "opencode_native_replay_plugin.js"
    )
    script = f"""
      import * as pluginModule from {plugin.as_uri()!r}
      const exports = Object.keys(pluginModule)
      if (exports.length !== 1 || exports[0] !== "default") process.exit(1)
    """
    subprocess.run([bun, "-e", script], check=True)


def test_running_bash_event_registers_dispatch_and_tool_without_completion(
    tmp_path: Path,
) -> None:
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is not installed")
    plugin = (
        Path(__file__).resolve().parents[1]
        / "minireplay"
        / "assets"
        / "opencode_native_replay_plugin.js"
    )
    gate = tmp_path / "start.gate"
    gate.write_text(
        '{"run_id":"run-test","opened_at_ns_decimal":"0"}\n',
        encoding="utf-8",
    )
    ready = tmp_path / "ready"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = f"""
      import NativeReplayPlugin from {plugin.as_uri()!r}
      const calls = []
      globalThis.fetch = async (url, options) => {{
        const payload = JSON.parse(options.body)
        calls.push({{url: String(url), payload}})
        const isDispatch = payload.kind === "dispatch"
        return {{
          ok: true,
          json: async () => ({{
            reservation_id: isDispatch ? "reservation-dispatch" : "reservation-tool",
            record_id: isDispatch ? "dispatch-0" : "tool-0",
            span_id: isDispatch ? "span-dispatch" : "span-tool",
          }}),
        }}
      }}
      const hooks = await NativeReplayPlugin({{
        client: {{}},
        directory: {str(workspace)!r},
      }})
      await hooks.event({{
        event: {{
          type: "message.part.updated",
          properties: {{
            part: {{
              type: "tool",
              tool: "bash",
              callID: "call-bash",
              sessionID: "child-session",
              state: {{status: "running", input: {{command: "./solution"}}}},
            }},
          }},
        }},
      }})
      if (calls.length !== 2) process.exit(1)
      if (calls[0].payload.kind !== "dispatch") process.exit(2)
      if (calls[1].payload.kind !== "tool") process.exit(3)
      if (calls.some((call) => call.url.endsWith("/complete"))) process.exit(4)
      if (calls[1].payload.parent_span_id !== "span-dispatch") process.exit(5)
      if (calls[0].payload.session_id !== "actor-3/invocation-0/root-0") process.exit(6)
    """
    env = {
        **os.environ,
        "NATIVE_REPLAY_BOUNDARY_URL": "http://boundary.test",
        "NATIVE_REPLAY_BOUNDARY_TOKEN": "token",
        "NATIVE_REPLAY_ACTOR_ID": "actor-3",
        "NATIVE_REPLAY_INVOCATION_ID": "actor-3/invocation-0",
        "NATIVE_REPLAY_READY_DIR": str(ready),
        "NATIVE_REPLAY_START_GATE": str(gate),
        "NATIVE_REPLAY_RUN_ID": "run-test",
        "NATIVE_REPLAY_GATE_ACTORS": "[]",
        "NATIVE_REPLAY_ARRIVAL_OFFSETS": "{}",
        "NATIVE_REPLAY_OPENCODE_IDENTITY": "opencode:test",
    }
    subprocess.run([bun, "-e", script], check=True, env=env)


def test_child_session_backfills_its_running_task_from_opencode_store(
    tmp_path: Path,
) -> None:
    bun = shutil.which("bun")
    if bun is None:
        pytest.skip("Bun is not installed")
    plugin = (
        Path(__file__).resolve().parents[1]
        / "minireplay"
        / "assets"
        / "opencode_native_replay_plugin.js"
    )
    gate = tmp_path / "start.gate"
    gate.write_text(
        '{"run_id":"run-test","opened_at_ns_decimal":"0"}\n',
        encoding="utf-8",
    )
    ready = tmp_path / "ready"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = f"""
      import NativeReplayPlugin from {plugin.as_uri()!r}
      const calls = []
      globalThis.fetch = async (url, options) => {{
        const payload = JSON.parse(options.body)
        calls.push({{url: String(url), payload}})
        const isDispatch = payload.kind === "dispatch"
        return {{
          ok: true,
          json: async () => ({{
            reservation_id: isDispatch ? "reservation-dispatch" : "reservation-tool",
            record_id: isDispatch ? "dispatch-0" : "tool-0",
            span_id: isDispatch ? "span-dispatch" : "span-tool",
          }}),
        }}
      }}
      const part = {{
        type: "tool",
        tool: "task",
        callID: "call-task",
        sessionID: "parent",
        state: {{
          status: "running",
          input: {{description: "research"}},
          metadata: {{sessionId: "child"}},
        }},
      }}
      const client = {{
        session: {{
          messages: async () => ({{data: [{{parts: [part]}}]}}),
        }},
      }}
      const hooks = await NativeReplayPlugin({{
        client,
        directory: {str(workspace)!r},
      }})
      await hooks.event({{
        event: {{
          type: "session.created",
          properties: {{info: {{id: "child", parentID: "parent"}}}},
        }},
      }})
      if (calls.length !== 2) process.exit(1)
      if (calls[0].payload.name !== "task") process.exit(2)
      const output = {{headers: {{}}}}
      await hooks["chat.headers"]({{sessionID: "child"}}, output)
      if (output.headers["X-Native-Replay-Session"] !==
          "actor-3/invocation-0/root-0/child-0") process.exit(3)
      if (output.headers["X-Native-Replay-Role"] !== "coral-subagent") process.exit(4)
      if (output.headers["X-Native-Replay-Parent-Span"] !== "span-tool") process.exit(5)
    """
    env = {
        **os.environ,
        "NATIVE_REPLAY_BOUNDARY_URL": "http://boundary.test",
        "NATIVE_REPLAY_BOUNDARY_TOKEN": "token",
        "NATIVE_REPLAY_ACTOR_ID": "actor-3",
        "NATIVE_REPLAY_INVOCATION_ID": "actor-3/invocation-0",
        "NATIVE_REPLAY_READY_DIR": str(ready),
        "NATIVE_REPLAY_START_GATE": str(gate),
        "NATIVE_REPLAY_RUN_ID": "run-test",
        "NATIVE_REPLAY_GATE_ACTORS": "[]",
        "NATIVE_REPLAY_ARRIVAL_OFFSETS": "{}",
        "NATIVE_REPLAY_OPENCODE_IDENTITY": "opencode:test",
    }
    subprocess.run([bun, "-e", script], check=True, env=env)
