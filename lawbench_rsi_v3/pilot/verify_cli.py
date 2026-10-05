"""Exercise the real local Claude CLI against a deterministic loopback fixture.

No credentials are loaded and no model provider is contacted. This verifies
tool execution and the guard integration, which pure Python unit tests cannot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot.state import ROOT, atomic_json, read_jsonl, utc_now
from pilot.proposer_contract import cli_environment


def run_check():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = ROOT / "external/cli_acceptance" / stamp
    package = output / "workspace"
    (package / ".prototypes").mkdir(parents=True)
    (package / "allowed.txt").write_text("LOCAL_ALLOWED_CANARY\n", encoding="utf-8")
    (package / "config.yaml").write_text("unchanged: true\n", encoding="utf-8")
    (output / "outside.txt").write_text("OUTSIDE_READ_MUST_BE_DENIED\n", encoding="utf-8")
    (package / ".prototypes/probe.py").write_text("print('PROTOTYPE_EXECUTED')\n", encoding="utf-8")
    operations = [
        ("Read", {"file_path": str(package / "allowed.txt")}),
        ("Glob", {"pattern": "*.txt", "path": str(package)}),
        ("Read", {"file_path": str(output / "outside.txt")}),
        ("Write", {"file_path": str(package / "config.yaml"), "content": "changed: true\n"}),
        ("Bash", {"command": "python .prototypes/probe.py", "description": "Run the offline prototype"}),
    ]
    # Exercise the actual LawBench item size and reading format, not only a
    # tiny synthetic text file. The source is allowed score/feedback, never audit.
    from pilot.diagnostics import materialize
    actual_sources = {
        "score": ROOT / "external/noise/R4B/1/score",
        "feedback": ROOT / "external/diagnostic_acceptance_source/feedback",
    }
    active_feedback = ROOT / "runs/D100_a/text_classification/history/confusion_disambiguation_memory/feedback"
    if (active_feedback / "complete.json").exists():
        actual_sources["feedback"] = active_feedback
    expected_items = {}
    for phase, source in actual_sources.items():
        if not (source / f"{phase}_traces.jsonl").exists():
            raise RuntimeError(f"Real {phase} trace source required for CLI acceptance")
        materialize(source, phase)
        destination = package / f"{phase}_diagnostics.jsonl"
        shutil.copyfile(source / "diagnostics.jsonl", destination)
        expected_items[phase] = [r["item_id"] for r in read_jsonl(destination)[:5]]
        operations.append(("Read", {"file_path": str(destination), "limit": 5}))
    score_row = read_jsonl(actual_sources["score"] / "diagnostics.jsonl")[0]
    shutil.copyfile(actual_sources["score"] / score_row["model_calls"][0]["prompt"], package / "actual_prompt.txt")
    operations.append(("Read", {"file_path": str(package / "actual_prompt.txt"), "limit": 10}))
    state = {"requests": 0, "tools": [], "results": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if "count_tokens" in self.path:
                payload = json.dumps({"input_tokens": 100}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if not self.path.startswith("/v1/messages"):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            state["tools"] = [t["name"] for t in body.get("tools", [])]
            for message in body.get("messages", []):
                content = message.get("content", [])
                for block in content if isinstance(content, list) else []:
                    if block.get("type") == "tool_result" and not any(r["tool_use_id"] == block["tool_use_id"] for r in state["results"]):
                        state["results"].append(block)
            index = state["requests"]
            state["requests"] += 1
            stop = "tool_use" if index < len(operations) else "end_turn"
            if index < len(operations):
                name, args = operations[index]
                block = {"type": "tool_use", "id": f"toolu_local_{index}", "name": name, "input": args}
            else:
                block = {"type": "text", "text": "LOCAL_FIXTURE_COMPLETE"}
            message = {"id": f"msg_local_{index}", "type": "message", "role": "assistant", "model": body.get("model"), "content": [block], "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 30}}
            if not body.get("stream"):
                payload = json.dumps(message).encode()
                content_type = "application/json"
            else:
                start = {**message, "content": [], "stop_reason": None, "usage": {"input_tokens": 100, "output_tokens": 0}}
                empty = {**block, "input": {}} if block["type"] == "tool_use" else {"type": "text", "text": ""}
                delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])} if block["type"] == "tool_use" else {"type": "text_delta", "text": block["text"]}
                events = [("message_start", {"type": "message_start", "message": start}),
                          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": empty}),
                          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 30}}),
                          ("message_stop", {"type": "message_stop"})]
                payload = "".join(f"event: {kind}\ndata: {json.dumps(value)}\n\n" for kind, value in events).encode()
                content_type = "text/event-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    spec = importlib.util.spec_from_file_location("cli_fixture_wrapper", ROOT / "engine/text_classification/claude_wrapper.py")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    original_command = wrapper.build_command
    def diagnostic_command(*args, **kwargs):
        return original_command(*args, **kwargs) + ["--include-hook-events", "--debug-file", str(output / "debug.log")]
    wrapper.build_command = diagnostic_command
    settings = {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": sys.executable.replace("\\", "/"), "args": [str(ROOT / "pilot/access_guard.py").replace("\\", "/")], "timeout": 15}]}]}}
    atomic_json(output / "settings.json", settings)
    env = {k: v for k, v in os.environ.items() if not any(word in k.upper() for word in ["KEY", "TOKEN", "SECRET", "AUTH", "ANTHROPIC", "OPENROUTER", "CLAUDE"])}
    env.update({"ANTHROPIC_AUTH_TOKEN": "local-fixture-no-real-key", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "PILOT_ALLOWED_ROOT": str(package), "PILOT_FROZEN_CODE": "[]", "PILOT_TOOL_ACCESS_LOG": str(output / "tool_access.jsonl"),
                "GIT_CEILING_DIRECTORIES": str(package.parent), "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
    cli_environment(env)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{server.server_port}"
    try:
        result = wrapper.run("Run the local acceptance fixture.", model="claude-opus-5-5-code", allowed_tools=["Read", "Glob", "Grep", "Write", "Edit", "Bash"],
                             cwd=str(package), settings_path=output / "settings.json", execution_env=env, log_dir=str(output / "sessions"),
                             name="local_fixture", timeout_seconds=90, progress=False)
    finally:
        server.shutdown()
        server.server_close()
    outcomes = {r["tool_use_id"]: r for r in state["results"]}
    def contains(index, text):
        return text in json.dumps(outcomes.get(f"toolu_local_{index}", {}), ensure_ascii=False)
    guard = read_jsonl(output / "tool_access.jsonl")
    checks = {"has_bash_tool": "Bash" in state["tools"],
              "read_allowed": contains(0, "LOCAL_ALLOWED_CANARY"),
              "glob_operational": contains(1, "allowed.txt") and not outcomes.get("toolu_local_1", {}).get("is_error"),
              "outside_read_denied": not contains(2, "OUTSIDE_READ_MUST_BE_DENIED") and bool(outcomes.get("toolu_local_2", {}).get("is_error")),
              "framework_write_denied": (package / "config.yaml").read_text(encoding="utf-8") == "unchanged: true\n",
              "prototype_executed": contains(4, "PROTOTYPE_EXECUTED"),
              "guard_executed": len(guard) >= 4 and any(r["decision"] == "deny" for r in guard),
              "real_score_rows_readable": all(contains(5, item) for item in expected_items["score"]) and not outcomes.get("toolu_local_5", {}).get("is_error"),
              "real_feedback_rows_readable": all(contains(6, item) for item in expected_items["feedback"]) and not outcomes.get("toolu_local_6", {}).get("is_error"),
              "actual_prompt_readable": bool(outcomes.get("toolu_local_7")) and not outcomes.get("toolu_local_7", {}).get("is_error"),
              "fixture_finished": "LOCAL_FIXTURE_COMPLETE" in result.text and result.exit_code == 0}
    report = {"created_at_utc": utc_now(), "passed": all(checks.values()), "checks": checks, "requests_local_only": state["requests"], "tools": state["tools"],
              "actual_model_calls": 0, "guard_events": guard, "tool_results": state["results"], "log_dir": str(output), "stderr": result.stderr}
    atomic_json(output / "receipt.json", report)
    print(json.dumps({k: report[k] for k in ["passed", "checks", "requests_local_only", "tools", "log_dir", "stderr"]}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    raise SystemExit(not run_check()["passed"])
