"""Visible history index and actual CLI execution acceptance for the pilot."""
from __future__ import annotations

import os
import ast
import re
import io
import tokenize
import shlex
import shutil
from pathlib import Path


def inspect_generality(source):
    """Enforce the existing skill's ban on task-specific hardcoded hints.

    No held-out answers or candidate scores are consulted. Documentation about
    the mechanism is excluded; executable crime-label/format hints are rejected.
    """
    tree = ast.parse(source)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            if "罪" in node.value:
                raise ValueError(f"Hardcoded task-specific hint in executable string at line {node.lineno}; violates the original general-purpose skill constraint")
    if re.search(r"\bLawBench\b", source, re.I):
        raise ValueError("The original skill forbids dataset names in candidate code, prompts and comments")


def normalize_documentation(source):
    """Remove a benchmark name only from comments/docstrings, never prompts.

    Keep the complete original source externally and verify that the AST after
    excluding documentation is identical. This repairs a documentation rule;
    it does not tune the generated mechanism or change an executable string.
    """
    if not re.search(r"\bLawBench\b", source, re.I):
        return source
    tree = ast.parse(source)
    if any(isinstance(node,ast.Name) and node.id == "__doc__" or isinstance(node,ast.Attribute) and node.attr == "__doc__" for node in ast.walk(tree)):
        return source
    positions = set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                positions.add((first.value.lineno,first.value.col_offset))
    tokens=[]
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT or token.type == tokenize.STRING and token.start in positions:
            token=token._replace(string=re.sub(r"\bLawBench\b","the benchmark",token.string,flags=re.I))
        tokens.append(token)
    updated=tokenize.untokenize(tokens)
    def executable_tree(text):
        parsed=ast.parse(text)
        for node in ast.walk(parsed):
            if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body:
                first=node.body[0]
                if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                    node.body=node.body[1:]
        return ast.dump(parsed,include_attributes=False)
    if executable_tree(source) != executable_tree(updated):
        raise ValueError("Documentation cleanup changed executable code")
    return updated


def cli_environment(env):
    """Resolve the installed Windows Git Bash; do not install or change tools."""
    git = shutil.which("git", path=env.get("PATH"))
    if os.name == "nt" and git:
        bash = Path(git).parent.parent / "bin/bash.exe"
        if bash.is_file():
            env["CLAUDE_CODE_GIT_BASH_PATH"] = str(bash)
    env.update({"USE_BUILTIN_RIPGREP": "0", "DISABLE_AUTOUPDATER": "1"})
    return env


def history_index(package):
    lines = []
    for folder in sorted((package / "history").iterdir()):
        for rel in ["train/log.jsonl", "train/diagnostics.jsonl", "score/diagnostics.jsonl", "feedback/diagnostics.jsonl"]:
            path = folder / rel
            if path.exists() and path.stat().st_size:
                count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
                lines.append(f"- {path.relative_to(package).as_posix()} ({count} records)")
    return "\n".join(lines)


def verify_execution(result, guard_events, package, candidates, prior_execution=None):
    init = next((event for event in result.raw_events if event.get("type") == "system" and event.get("subtype") == "init"), {})
    if "Bash" not in init.get("tools", []):
        raise RuntimeError("CLI has no Bash tool; prototypes cannot execute")
    guarded = {event.get("tool_use_id") for event in guard_events if event["decision"] == "allow"}
    successful = [tool for tool in result.tool_calls if not tool.is_error]
    unguarded = [tool.tool_id for tool in successful if tool.tool_id not in guarded]
    if unguarded:
        raise RuntimeError("Successful tools lack verified PreToolUse guard records")
    empirical = []
    for tool in successful:
        if tool.name in {"Read", "Grep"}:
            value = tool.input.get("file_path") or tool.input.get("path") or ""
            path = Path(value)
            if not path.is_absolute():
                path = package / path
            try:
                relative = path.resolve().relative_to(package.resolve()).as_posix()
            except ValueError:
                raise RuntimeError("CLI read outside the permitted workspace")
            if tool.output and tool.output.strip() != "No matches found" and relative.startswith("history/"):
                empirical.append(relative)
    current_empirical = sorted(set(empirical))
    if prior_execution:
        # A resumed iteration can reuse its own verified native history reads.
        # Prototype and guarded-tool checks below still apply to this session.
        assert prior_execution["passed"]
        empirical.extend(prior_execution["empirical_paths_read"])
    if not any(p.endswith(("log.jsonl", "diagnostics.jsonl", "train_traces.jsonl", "score_traces.jsonl", "feedback_traces.jsonl", "calls.jsonl")) for p in empirical):
        raise RuntimeError("No successful per-item empirical-log read")
    # Availability is fixed by the controller. Actual use is an observation,
    # not a criterion for repeatedly generating a preferred experimental sample.
    score_read = any("/score/" in p for p in empirical)
    executed = set()
    for tool in successful:
        if tool.name == "Bash":
            parts = shlex.split(tool.input.get("command", ""))
            if len(parts) == 2:
                script = Path(parts[1])
                if not script.is_absolute():
                    script = package / script
                executed.add(script.resolve().relative_to(package.resolve()).as_posix())
    # A prototype may carry a role suffix without changing its association
    # with the candidate. Keep the directory and full candidate name exact.
    candidate_prototypes = {}
    for candidate in candidates:
        name = candidate["name"]
        accepted_names = {name, name + "_proto", name + "_prototype", name + "_final", "test_" + name}
        candidate_prototypes[name] = sorted(p for p in executed
            if Path(p).parent.as_posix() == ".prototypes" and Path(p).suffix == ".py" and Path(p).stem in accepted_names)
    if any(not paths for paths in candidate_prototypes.values()):
        raise RuntimeError("Each final candidate needs a successfully executed matching prototype")
    return {"passed": True, "successful_guarded_tools": len(successful), "empirical_paths_read": sorted(set(empirical)),
            "current_session_empirical_paths_read": current_empirical, "reused_native_execution": prior_execution,
            "prototype_commands_executed": sorted(executed), "candidate_prototypes": candidate_prototypes, "score_read_observed": score_read,
            "feedback_read_observed": any("/feedback/" in p for p in empirical)}
