"""Claude PreToolUse allowlist. Operational isolation, not an OS sandbox.

Hook schema: https://code.claude.com/docs/en/hooks
"""
from __future__ import annotations

import ast
import json
import os
import shlex
import sys
from pathlib import Path

FORBIDDEN_IMPORTS = {"os", "sys", "pathlib", "subprocess", "socket", "requests", "urllib", "http", "ctypes", "importlib", "inspect", "builtins", "io", "ssl", "dotenv", "pickle", "marshal", "shelve", "litellm", "anthropic", "openai"}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "breakpoint", "input", "LLM", "ProviderLLM"}
FORBIDDEN_ATTRS = {"read_text", "read_bytes", "write_text", "write_bytes", "open", "loadtxt", "savetxt", "fromfile", "tofile", "system", "popen", "getenv", "environ", "__globals__", "__builtins__", "__subclasses__", "LLM", "ProviderLLM", "propose_claude", "run_benchmark", "load_pilot_split"}


def inspect_python(text: str):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in FORBIDDEN_IMPORTS for a in node.names):
                raise ValueError("Prototype/candidate may not access external files, processes, or network")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                raise ValueError("External I/O import is not permitted")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError("Use the injected LLM interface; no direct I/O or dynamic execution")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise ValueError("External I/O or runtime introspection is not permitted")
    return tree


def inside(root: Path, value: str) -> Path:
    path = Path(value)
    target = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not target.is_relative_to(root):
        raise ValueError("This tool may only access this run's workspace")
    if any(p in {".env", ".git", ".claude"} for p in target.relative_to(root).parts):
        raise ValueError("Configuration and credentials are not accessible")
    return target


def check_tool(root: Path, frozen: set[str], event: dict):
    tool, args = event.get("tool_name"), event.get("tool_input", {})
    if tool in {"Read", "Glob", "Grep"}:
        path = inside(root, args.get("file_path") or args.get("path") or ".")
        if os.environ.get("PILOT_INJECTED_RUN") == "1" and tool == "Read":
            rel = path.relative_to(root).as_posix()
            if rel.endswith("/feedback/diagnostics.jsonl") or rel.endswith("/feedback/feedback_traces.jsonl") or rel.endswith("/feedback/calls.jsonl"):
                raise ValueError("All 100 feedback records are already embedded in the initial prompt. Use that complete block; do not duplicate it. Individual details remain readable.")
            if rel.startswith("history/") and path.suffix == ".jsonl" and (not args.get("limit") or int(args["limit"]) > 5):
                raise ValueError("Read history JSONL in chunks with limit<=5 to preserve room for implementation; use offset for another chunk.")
            if rel.startswith("history/") and path.suffix in {".json", ".jsonl"} and path.is_file():
                lines = path.read_text(encoding="utf-8").splitlines()
                offset = max(0, int(args.get("offset", 1)) - 1)
                limit = int(args.get("limit") or len(lines))
                if sum(len(line) for line in lines[offset:offset+limit]) > 8000:
                    raise ValueError("This slice contains a huge serialized state. Use train/diagnostics.jsonl for step records or smaller slices of train/memory.json; do not duplicate checkpoint blobs.")
        if tool == "Glob" and (".." in args.get("pattern", "") or Path(args.get("pattern", "")).is_absolute()):
            raise ValueError("Glob must remain inside this run")
        return str(path)
    if tool in {"Write", "Edit"}:
        path = inside(root, args["file_path"])
        rel = path.relative_to(root).as_posix()
        if rel in frozen:
            raise ValueError("Previously evaluated candidate code is immutable")
        if not (rel.startswith("agents/") and path.suffix == ".py" or rel.startswith(".prototypes/") and path.suffix == ".py" or rel.startswith("reports/") or rel == "pending_eval.json"):
            raise ValueError("Only new candidates, offline prototypes, reports, and pending_eval.json may be written")
        return rel
    if tool == "Bash":
        command = args.get("command", "")
        if any(t in command for t in [";", "&&", "||", "|", ">", "<", "`", "$", "\n"]):
            raise ValueError("Use one offline Python prototype command without shell operators")
        parts = shlex.split(command)
        if len(parts) != 2 or Path(parts[0]).name.lower() not in {"python", "python.exe"}:
            raise ValueError("Allowed command: python .prototypes/<name>.py")
        script = inside(root, parts[1])
        if script.parent != root / ".prototypes" or script.suffix != ".py":
            raise ValueError("Only local offline prototypes may execute")
        inspect_python(script.read_text(encoding="utf-8"))
        for source in (root / "agents").glob("*.py"):
            inspect_python(source.read_text(encoding="utf-8"))
        return str(script)
    raise ValueError("Tool is outside this pilot's allowlist")


def main():
    event = json.load(sys.stdin)
    root = Path(os.environ["PILOT_ALLOWED_ROOT"]).resolve()
    frozen = set(json.loads(os.environ["PILOT_FROZEN_CODE"]))
    try:
        target = check_tool(root, frozen, event)
        decision, reason = "allow", "Allowed pilot operation"
    except Exception as exc:
        target = None
        decision, reason = "deny", str(exc)
    audit = Path(os.environ["PILOT_TOOL_ACCESS_LOG"])
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": event.get("tool_name"), "tool_use_id": event.get("tool_use_id"), "target": target, "decision": decision}, ensure_ascii=False) + "\n")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": decision, "permissionDecisionReason": reason}}))


if __name__ == "__main__":
    main()
