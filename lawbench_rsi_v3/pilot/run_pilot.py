"""Ordered and resumable controller. Run with the original project Python 3.11."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot.state import (ROOT, Ledger, atomic_json, digest, exclusive_controller, file_lock, finish, jsonl_write,
                         object_hash, read_json, read_jsonl, utc_now, verify_done, write_once)
from pilot.access_guard import inspect_python
from pilot.proposer_contract import cli_environment, history_index, verify_execution
from pilot.diagnostics import materialize

ENGINE = ROOT / "engine" / "text_classification"
BASE = "confusion_disambiguation_memory"
FROZEN = {"R4B": BASE, "R7A": "adaptive_tokenizer_confusion_memory", "R12B": "discriminative_similarity_memory"}
CORE_RUNS = ["D0_a", "D0_b", "D100_a", "D100_b"]
MAIN_PHASES = {"train", "score", "feedback", "proposer"}


def config():
    return read_json(ROOT / "pilot_config.json")


def package_for(run_id):
    return ROOT / "runs" / run_id / "text_classification"


def control_for(run_id):
    return ROOT / "external" / "control" / run_id


def engine_inventory():
    paths = list(ENGINE.glob("*.py")) + list((ENGINE / "data").glob("*.py"))
    paths += [ENGINE / "config.yaml", ENGINE / ".claude/skills/meta-harness/SKILL.md"]
    return {p.relative_to(ENGINE).as_posix(): digest(p) for p in paths}


def create_run(run_id):
    cfg = config()
    if run_id not in cfg["runs"]:
        raise ValueError("Unknown run")
    if run_id == "D50_c":
        if 4 not in cfg.get("enabled_experiments", [1, 2, 3, 4]):
            raise PermissionError("Experiment 4 is deferred by the user")
        from pilot.analyze import verify_prediction
        verify_prediction()
    package = package_for(run_id)
    control = control_for(run_id)
    marker = control / "initialized.json"
    fingerprint = {"framework": engine_inventory(), "manifest": digest(ROOT / cfg["manifest"]), "D": cfg["runs"][run_id]}
    if marker.exists():
        saved_fingerprint = read_json(marker)["fingerprint"]
        if any(saved_fingerprint[k] != fingerprint[k] for k in ["manifest", "D"]):
            raise ValueError("Run inputs differ from initialization")
        # Existing runs keep their frozen framework, even after an additive
        # controller/CLI transport improvement for a later supplementary run.
        for rel, expected in saved_fingerprint["framework"].items():
            if digest(package / rel) != expected:
                raise ValueError("Run framework changed after initialization")
        return package
    if package.exists() and any(package.iterdir()):
        raise ValueError("Uninitialized run directory is not empty")
    for rel in fingerprint["framework"]:
        target = package / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ENGINE / rel, target)
    for name in ["__init__.py", BASE + ".py"]:
        target = package / "agents" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ENGINE / "agents" / name, target)
    for folder in ["reports", ".prototypes"]:
        (package / folder).mkdir(exist_ok=True)
    cache = ROOT / "external" / "caches" / run_id
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("New run cache must be empty")
    cache.mkdir(parents=True, exist_ok=True)
    source = (ROOT / cfg["old_run"]).resolve() / "LawBench" / BASE / "gpt-oss-120b"
    target = package / "history" / BASE / "train"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "memory.json", target / "memory.json")
    legal = [row for row in read_jsonl(source / "log.jsonl") if row.get("type") in {"meta", "step", "learn_batch", "checkpoint", "train_batch"}]
    jsonl_write(target / "log.jsonl", legal)
    jsonl_write(package / "evolution_summary.jsonl", [])
    atomic_json(package / "frontier_val.json", {})
    control.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, {"run_id": run_id, "fingerprint": fingerprint, "created_at_utc": utc_now(), "base_code_hash": digest(package / "agents" / (BASE + ".py")), "base_memory_hash": digest(target / "memory.json")})
    return package


def stage_output(run_id, candidate, phase):
    if phase == "audit":
        return ROOT / "external" / "audit" / run_id / candidate
    return package_for(run_id) / "history" / candidate / phase


def run_worker(spec):
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=True)
    spec_path = ROOT / "external" / "jobs" / (object_hash(spec) + ".json")
    if spec_path.exists():
        if read_json(spec_path) != spec:
            raise ValueError("Worker spec hash collision")
    else:
        atomic_json(spec_path, spec)
    env = os.environ.copy()
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_LOG": "ERROR"})
    # Evaluation uses the trusted worker; no benchmark command can discover other datasets.
    with (output / "worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.run([sys.executable, str(ROOT / "pilot/worker.py"), "--spec", str(spec_path)],
                                 cwd=spec["package"], env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Worker paused: {spec['run_id']}/{spec['candidate']}/{spec['phase']}; see {output / 'failure.json'}")
    return read_json(output / "result.json")


def run_stage(run_id, candidate, round_number, phase):
    cfg = config()
    if run_id in cfg.get("injected", {}).get("runs", []) and candidate == BASE:
        from pilot.injected import reused_stage
        return reused_stage(run_id, phase)
    package = package_for(run_id)
    output = stage_output(run_id, candidate, phase)
    memory = package / "history" / candidate / "train" / "memory.json"
    spec = {"run_id": run_id, "candidate": candidate, "round": round_number, "phase": phase,
            "D": cfg["runs"][run_id], "package": str(package), "output": str(output),
            "memory": str(memory) if phase != "train" else None,
            "cache_mode": "run", "cache_dir": str(ROOT / "external/caches" / run_id)}
    result = run_worker(spec)
    if phase in {"train", "score", "feedback"}:
        materialize(output, phase)
    return result


def main_cost_snapshot(run_id):
    ledger = read_jsonl(ROOT / "external/cost_ledger.jsonl")
    rows = [r for r in ledger if r.get("run_id") == run_id and r.get("phase") in MAIN_PHASES]
    paid = [r for r in rows if r["event"] in {"api_result", "proposer_result"}]
    recovery_path = control_for(run_id) / "recovery.json"
    recovery = read_json(recovery_path) if recovery_path.exists() else {}
    failed_ids = set(recovery.get("failed_setup_event_ids", []))
    failed_cost = sum(r.get("cost") or 0 for r in paid if r["event_id"] in failed_ids)
    closed = {r["attempt_id"] for r in rows if r["event"] in {"api_result", "proposer_result"} or r["event"] == "api_failed" and r.get("http_status") == 402}
    unknown = [r for r in rows if r["event"] in {"api_started", "proposer_started"} and r["attempt_id"] not in closed]
    return {"cost_usd": sum(r.get("cost") or 0 for r in paid),
            "failed_setup_cost_usd": failed_cost,
            "experimental_cost_usd": sum(r.get("cost") or 0 for r in paid) - failed_cost,
            "reserved_or_unknown_usd": sum(r["reserved_usd"] for r in unknown),
            "unknown_attempt_ids": [r["attempt_id"] for r in unknown],
            "ledger_event_ids": [r["event_id"] for r in rows],
            "cost_events_hash": object_hash(rows),
            "unknown_failed_attempts": len(unknown),
            "nonbillable_credit_rejections": sum(r["event"] == "api_failed" and r.get("http_status") == 402 for r in rows),
            "captured_at_utc": utc_now()}


def select_earliest(rows):
    valid = [r for r in rows if r.get("correct") is not None]
    if not valid:
        raise ValueError("No valid scored candidate")
    return max(valid, key=lambda r: r["correct"])


def save_checkpoint(run_id, round_number, ordered_candidates):
    path = control_for(run_id) / f"round{round_number}_checkpoint.json"
    if path.exists():
        checkpoint = read_json(path)
        for p, sha in checkpoint["artifacts"].items():
            if digest(ROOT / p) != sha:
                raise ValueError("Checkpoint artifact changed")
        return checkpoint
    rows = []
    artifacts = {}
    for candidate in ordered_candidates:
        result_path = stage_output(run_id, candidate, "score") / "result.json"
        invalid = control_for(run_id) / "invalid" / (candidate + ".json")
        if invalid.exists():
            rows.append({"candidate": candidate, "correct": None, "status": "invalid"})
            artifacts[invalid.relative_to(ROOT).as_posix()] = digest(invalid)
            continue
        result = read_json(result_path)
        rows.append({"candidate": candidate, "correct": result["correct"], "total": result["total"], "status": "valid"})
        for p in [result_path, package_for(run_id) / "agents" / (candidate + ".py"), package_for(run_id) / "history" / candidate / "train/memory.json"]:
            artifacts[p.relative_to(ROOT).as_posix()] = digest(p)
    selected = select_earliest(rows)["candidate"]
    checkpoint = {"run_id": run_id, "T": round_number, "N": 2 * round_number,
                  "D": config()["runs"][run_id], "selected": selected, "candidates": rows,
                  "artifacts": artifacts, "cost": main_cost_snapshot(run_id), "created_at_utc": utc_now()}
    write_once(path, checkpoint)
    atomic_json(package_for(run_id) / "frontier_val.json", {"selected": selected, "score_correct": select_earliest(rows)["correct"], "score_total": 100, "candidate_order": rows})
    return checkpoint


def validate_candidate(package, name):
    source = package / "agents" / (name + ".py")
    inspect_python(source.read_text(encoding="utf-8"))
    if not config().get("allow_task_format_hints", False):
        from pilot.proposer_contract import inspect_generality
        inspect_generality(source.read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / "pilot/validate_candidate.py"), str(package), name]
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in ["API_KEY", "TOKEN", "SECRET", "AUTH", "ANTHROPIC", "OPENROUTER"])}
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"})
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, encoding="utf-8")
    if result.returncode:
        raise ValueError("Candidate failed the fixed offline interface acceptance")


def propose(run_id, round_number):
    cfg = config()
    if not cfg["budget_confirmed"]:
        raise PermissionError("Budget requires user confirmation")
    package = package_for(run_id)
    out = control_for(run_id) / f"proposer_round{round_number}"
    out.mkdir(parents=True, exist_ok=True)
    signature = {"run_id": run_id, "round": round_number, "D": cfg["runs"][run_id], "model": cfg["proposer"], "effort": cfg["proposer_effort"]}
    if (out / "complete.json").exists():
        verify_done(out, signature)
        pending = read_json(out / "pending_eval.json")
        for row in pending["candidates"]:
            if digest(package / row["file"]) != row["code_hash"]:
                raise ValueError("Generated candidate was modified")
        return pending["candidates"]
    if (out / "started.json").exists():
        if (out / "result.json").exists():
            started = read_json(out / "started.json")
            if started["signature"] != signature:
                raise ValueError("Saved proposer inputs changed")
            # Revalidate the already generated artifacts after an interrupted
            # acceptance step. This branch never calls the proposer again.
            wrapper_spec = importlib.util.spec_from_file_location("saved_proposer_wrapper", ENGINE / "claude_wrapper.py")
            wrapper = importlib.util.module_from_spec(wrapper_spec)
            sys.modules[wrapper_spec.name] = wrapper
            wrapper_spec.loader.exec_module(wrapper)
            sessions = list((out / "sessions").glob("*/meta.json"))
            if len(sessions) != 1:
                raise RuntimeError("Expected exactly one retained proposer session")
            meta = read_json(sessions[0])
            result = wrapper.parse_stream_events((sessions[0].parent / "events.jsonl").read_text(encoding="utf-8"),
                                                  meta["prompt"], meta["model"], meta["duration_seconds"], meta["exit_code"], str(package))
            return accept_completed_proposal(run_id, round_number, result, read_json(out / "result.json"), started["frozen_code"], signature)
        raise RuntimeError("Prior proposer session is incomplete; retained artifacts require review, not regeneration")
    frozen = {f"agents/{p.name}": digest(p) for p in (package / "agents").glob("*.py")}
    if run_id in cfg.get("injected", {}).get("runs", []):
        # Unfinished files from the same interrupted proposal are editable;
        # H0 and anything that has entered evaluation remain immutable.
        frozen = {rel:sha for rel,sha in frozen.items() if Path(rel).stem in {BASE, "__init__"}
                  or (package / "history" / Path(rel).stem).exists()}
    pending_path = package / "pending_eval.json"
    if pending_path.exists():
        # Preserve the previous round's file in the controller before the next write.
        shutil.copyfile(pending_path, out / "previous_pending_eval.json")
    wrapper_spec = importlib.util.spec_from_file_location("pilot_claude_wrapper", package / "claude_wrapper.py")
    wrapper = importlib.util.module_from_spec(wrapper_spec)
    sys.modules[wrapper_spec.name] = wrapper
    wrapper_spec.loader.exec_module(wrapper)
    # Import just the prompt function, without executing the original evolution loop.
    import ast
    tree = ast.parse((ENGINE / "meta_harness.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_task_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pilot prompt>", "exec"), namespace)
    prompt = namespace["render_task_prompt"](round_number, 1, {"run_id": run_id, "D": cfg["runs"][run_id], "history_index": history_index(package)})
    injection_guard = None
    if run_id in cfg.get("injected", {}).get("runs", []):
        from pilot.injected import injected_prompt, budget_guard
        prompt = injected_prompt(run_id, prompt)
        injection_guard = budget_guard(run_id)
    settings = {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": sys.executable.replace("\\", "/"), "args": [str(ROOT / "pilot/access_guard.py").replace("\\", "/")], "timeout": 15}]}]}}
    settings_path = out / "settings.json"
    atomic_json(settings_path, settings)
    from dotenv import dotenv_values
    credentials = dotenv_values((ROOT / cfg["credentials_file"]).resolve())
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in ["API_KEY", "TOKEN", "SECRET", "AUTH", "ANTHROPIC", "OPENROUTER", "CLAUDECODE"])}
    env.update({k: credentials[k] for k in ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"]})
    env.update({"PYTHONPATH": str(package.parent), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
                "TEXT_CLASSIFICATION_LLM_CACHE_MODE": "off", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PILOT_ALLOWED_ROOT": str(package), "PILOT_FROZEN_CODE": json.dumps(list(frozen)),
                "PILOT_TOOL_ACCESS_LOG": str(out / "tool_access.jsonl"),
                "GIT_CEILING_DIRECTORIES": str(package.parent), "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
    cli_environment(env)
    if injection_guard:
        env.pop("DISABLE_AUTO_COMPACT", None)
        env.pop("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", None)
    ledger = Ledger(ROOT / "external/cost_ledger.jsonl", cfg["budget_usd"])
    cost = ledger.summary()
    allowance = (cfg["budget_usd"] - cost["known_or_estimated_usd"] - cost["reserved_or_unknown_usd"]) / cfg["concurrent_trajectories"]
    if injection_guard:
        injection_state = injection_guard.state()
        used = sum(r.get("cost_usd", r["reserved_usd"]) for r in injection_state["requests"].values())
        unfinished = sum(not (control_for(r) / "proposer_round1/complete.json").exists() for r in cfg["injected"]["runs"])
        allowance = max(0.0, (cfg["injected"]["proposer_budget_usd"] - used) / max(1, unfinished))
    if allowance <= 0 and not injection_guard:
        raise RuntimeError("Budget exhausted")
    recovery = read_json(control_for(run_id) / "recovery.json") if (control_for(run_id) / "recovery.json").exists() else {}
    attempt_id = f"{run_id}:proposer:{round_number}" + recovery.get("attempt_suffix", "")
    metadata = {**signature, "candidate": None, "phase": "proposer", "attempt_id": attempt_id}
    ledger.add({**metadata, "event": "proposer_started", "event_id": attempt_id + ":started", "reserved_usd": allowance})
    write_once(out / "started.json", {"signature": signature, "started_at_utc": utc_now(), "frozen_code": frozen})
    def observe_proposer(event, _tools):
        with (out / "live_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    from pilot.proposer_relay import ProposerRelay, summarize_requests
    with ProposerRelay(credentials["ANTHROPIC_BASE_URL"], out / "http_requests.jsonl", guard=injection_guard) as relay:
        env["ANTHROPIC_BASE_URL"] = relay.url
        result = wrapper.run(prompt=prompt, model=cfg["proposer"], effort=cfg["proposer_effort"],
                             allowed_tools=["Read", "Glob", "Grep", "Write", "Edit", "Bash"],
                             skills=[str(package / ".claude/skills/meta-harness")], cwd=str(package),
                             log_dir=str(out / "sessions"), name=f"round{round_number}",
                             timeout_seconds=cfg["proposer_timeout_seconds"], progress=observe_proposer,
                             settings_path=settings_path, max_budget_usd=None if injection_guard else allowance, execution_env=env)
    request_counts = summarize_requests(read_jsonl(out / "http_requests.jsonl"))
    result_events = [r for r in result.raw_events if r.get("type") == "result"]
    success = result.exit_code == 0 and bool(result_events) and not result_events[-1].get("is_error")
    cost_known = bool(result_events) and "total_cost_usd" in result_events[-1]
    message_ids = {r.get("message", {}).get("id") for r in result.raw_events if r.get("type") == "assistant" and r.get("message", {}).get("id")}
    record = {**metadata, **request_counts, "logical_calls": len(message_ids), "cache_hits": 0,
              "provider_usage": result.token_usage,
              "event": "proposer_result" if cost_known else "proposer_failed", "event_id": attempt_id + ":result",
              "cost": result.cost_usd if cost_known else None, "cost_source": "claude_cli_log_estimate", "input_tokens": result.token_usage.get("input_tokens", 0),
              "output_tokens": result.token_usage.get("output_tokens", 0), "wall_seconds": result.duration_seconds,
              "exit_code": result.exit_code, "session_id": result.session_id, "success": success}
    ledger.add(record)
    atomic_json(out / "result.json", record)
    return accept_completed_proposal(run_id, round_number, result, record, frozen, signature)


def accept_completed_proposal(run_id, round_number, result, record, frozen, signature):
    """Accept one saved session; never regenerate because of its design choices."""
    package = package_for(run_id)
    out = control_for(run_id) / f"proposer_round{round_number}"
    pending_path = package / "pending_eval.json"
    for rel, expected in frozen.items():
        if digest(package / rel) != expected:
            raise RuntimeError("Optimizer changed a frozen candidate; stop")
    for rel, expected in read_json(control_for(run_id) / "initialized.json")["fingerprint"]["framework"].items():
        if digest(package / rel) != expected:
            raise RuntimeError("Optimizer changed the fixed framework; stop")
    if not record["success"]:
        raise RuntimeError(f"Proposer failed for {run_id}/round{round_number}; session retained")
    if record["cost"] is None:
        raise RuntimeError("Proposer cost is unreported; cannot silently count it as zero")
    pending = read_json(pending_path)
    if pending.get("iteration") != round_number or len(pending.get("candidates", [])) != 2:
        raise ValueError("Proposer must produce exactly A/B for the requested round")
    prior_execution = None
    if run_id in config().get("injected", {}).get("runs", []):
        from pilot.injected import prior_native_execution
        prior_execution = prior_native_execution(run_id)
    execution = verify_execution(result, read_jsonl(out / "tool_access.jsonl"), package, pending["candidates"], prior_execution)
    atomic_json(out / "execution_acceptance.json", execution)
    names = []
    documentation_files = []
    for index, candidate in enumerate(pending["candidates"]):
        name = candidate["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names or f"agents/{name}.py" in frozen:
            raise ValueError("Invalid or reused candidate name")
        if candidate["file"] != f"agents/{name}.py":
            raise ValueError("Candidate file must match its name")
        from pilot.proposer_contract import normalize_documentation
        source_path = package / candidate["file"]
        generated_source = source_path.read_text(encoding="utf-8")
        normalized = generated_source if config().get("allow_task_format_hints", False) else normalize_documentation(generated_source)
        if normalized != generated_source:
            original_hash = digest(source_path)
            original_path = out / "original_generated_sources" / (name + ".py")
            original_path.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_path, original_path)
            source_path.write_text(normalized, encoding="utf-8")
            atomic_json(out / f"{name}_documentation_cleanup.json", {"generated_code_hash": original_hash, "executed_code_hash": digest(source_path), "executable_AST_unchanged": True, "scope": "benchmark name in comments/docstrings only"})
            candidate["generated_code_hash"] = original_hash
            documentation_files.extend([original_path.relative_to(out).as_posix(), f"{name}_documentation_cleanup.json"])
        candidate["code_hash"] = digest(package / candidate["file"])
        candidate["slot"] = "AB"[index]
        names.append(name)
        try:
            validate_candidate(package, name)
        except (ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
            invalid_record = {"run_id": run_id, "round": round_number, "candidate": name, "stage": "fixed_offline_acceptance", "error_type": type(exc).__name__, "reason": str(exc), "status": "invalid", "replacement_generated": False}
            atomic_json(control_for(run_id) / "invalid" / f"{name}.json", invalid_record)
            atomic_json(package / "reports" / f"invalid_{name}.json", invalid_record)
    feedback_reads = []
    for tool in result.tool_calls:
        path = tool.input.get("file_path") or tool.input.get("path") or ""
        if tool.name in {"Read", "Grep"} and "feedback" in path:
            feedback_reads.append({"tool": tool.name, "path": path, "item_ids_in_tool_output": sorted(set(re.findall(r"lawbench_3-3_\d{4}", str(tool.output)))), "is_error": tool.is_error})
    atomic_json(out / "feedback_access.json", {"observed_feedback_reads": feedback_reads, "note": "IDs in returned tool text measure visibility, not proof the model used every item"})
    atomic_json(out / "pending_eval.json", pending)
    finish(out, signature, ["result.json", "pending_eval.json", "feedback_access.json", "http_requests.jsonl", "execution_acceptance.json", "tool_access.jsonl"] + documentation_files)
    return pending["candidates"]


def round_run(run_id, round_number):
    create_run(run_id)
    if round_number == 1:
        run_stage(run_id, BASE, 0, "score")
        run_stage(run_id, BASE, 0, "feedback")
        ordered = [BASE]
        score = read_json(stage_output(run_id, BASE, "score") / "result.json")
        if not (control_for(run_id) / "round1_checkpoint.json").exists():
            atomic_json(package_for(run_id) / "frontier_val.json", {"selected": BASE, "score_correct": score["correct"], "score_total": 100})
    else:
        previous = read_json(control_for(run_id) / "round1_checkpoint.json")
        ordered = [r["candidate"] for r in previous["candidates"]]
        save_checkpoint(run_id, 1, ordered)
        for candidate in ordered[1:]:
            if not (control_for(run_id) / "invalid" / f"{candidate}.json").exists():
                run_stage(run_id, candidate, 1, "feedback")
    candidates = propose(run_id, round_number)
    summary_path = package_for(run_id) / "evolution_summary.jsonl"
    summary = read_jsonl(summary_path)
    for candidate in candidates:
        name = candidate["name"]
        ordered.append(name)
        if (control_for(run_id) / "invalid" / f"{name}.json").exists():
            continue
        run_stage(run_id, name, round_number, "train")
        result = run_stage(run_id, name, round_number, "score")
        if not any(r.get("system") == name for r in summary):
            summary.append({**candidate, "iteration": round_number, "system": name, "avg_val": result["accuracy"] * 100, "score_correct": result["correct"], "score_total": 100})
            jsonl_write(summary_path, summary)
    checkpoint = save_checkpoint(run_id, round_number, ordered)
    # All audit tasks follow the immutable selection/cost snapshot.
    for candidate in candidates:
        name = candidate["name"]
        if not (control_for(run_id) / "invalid" / f"{name}.json").exists():
            run_stage(run_id, name, round_number, "audit")
    return checkpoint


def two_at_a_time(function, items):
    results = []
    for offset in range(0, len(items), 2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(function, item) for item in items[offset:offset + 2]]
            for future in futures:
                results.append(future.result())
    return results


def noise():
    cfg = config()
    def evaluate(job):
        label, repetition = job
        candidate = FROZEN[label]
        memory = (ROOT / cfg["old_run"]).resolve() / "LawBench" / candidate / "gpt-oss-120b/memory.json"
        for phase in ["score", "audit"]:
            run_worker({"run_id": f"noise_{label}_{repetition}", "candidate": candidate, "round": 0, "D": 0,
                        "phase": phase, "package": str(ENGINE), "memory": str(memory), "cache_mode": "off",
                        "cache_dir": None, "output": str(ROOT / "external/noise" / label / str(repetition) / phase)})
    two_at_a_time(evaluate, [(label, rep) for label in FROZEN for rep in [1, 2]])
    from pilot.analyze import analyze_noise
    analyze_noise()


def preflight():
    cfg = config()
    manifest = read_json(ROOT / cfg["manifest"])
    audit = read_json((ROOT / cfg["manifest"]).parent / "data_audit.json")
    if audit["status"] != "PASS":
        raise ValueError("Data audit did not pass")
    for split in manifest["splits"].values():
        if digest((ROOT / cfg["manifest"]).parent / split["file"]) != split["sha256"]:
            raise ValueError("Prepared data changed")
    expected = {"solver": "openrouter/openai/gpt-oss-120b", "proposer": "claude-opus-5-5-code", "proposer_effort": "high", "temperature": 0.0, "max_tokens": 16384, "training_seed": 42, "split_seed": 20261003, "candidates_per_round": 2, "rounds": 2, "evaluation_workers": 16, "concurrent_trajectories": 2, "max_api_retries": 3}
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f"Locked setting changed: {key}")
    from pilot.verify import run_verification
    report = run_verification()
    if not report["passed"]:
        raise RuntimeError("Offline verification failed")
    print(json.dumps({"data_audit": "PASS", "offline_tests": "PASS", "budget_confirmed": cfg["budget_confirmed"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "noise", "round1", "round2", "predict", "holdout", "analyze", "all", "injected-prepare", "injected"], default="preflight")
    parser.add_argument("--resume", action="store_true", help="Completed stages are hash-checked and skipped; unfinished proposer sessions require review")
    args = parser.parse_args()
    if args.stage in {"injected-prepare", "injected"}:
        from pilot.injected import run_injected
        run_injected(paid=args.stage == "injected")
        return
    enabled = config().get("enabled_experiments", [1, 2, 3, 4])
    stage_experiment = {"noise": 1, "round1": 2, "round2": 3, "predict": 4, "holdout": 4}
    all_stages = ["preflight"] + [s for s,n in stage_experiment.items() if n in enabled] + ["analyze"]
    stages = all_stages if args.stage == "all" else [args.stage]
    if any(s in stage_experiment and stage_experiment[s] not in enabled for s in stages):
        raise SystemExit("Requested experiment is outside the user-authorized scope")
    if any(s in stages for s in ["noise", "round1", "round2", "holdout"]) and not config()["budget_confirmed"]:
        raise SystemExit("Paid execution pending user-confirmed budget in pilot_config.json")
    for stage in stages:
        if stage in {"noise", "round1", "round2", "holdout"}:
            verified = read_json(ROOT / "verification.json")
            if not verified["passed"]:
                raise RuntimeError("Complete offline acceptance first")
            for rel, expected in verified["source_hashes"].items():
                if digest(ROOT / rel) != expected:
                    raise ValueError("Pilot source changed after verification; rerun preflight")
        atomic_json(ROOT / "external/status.json", {"stage": stage, "state": "running", "updated_at_utc": utc_now()})
        try:
            if stage in {"noise", "round1", "round2", "holdout"}:
                from pilot.billing import ensure_solver_credits
                ensure_solver_credits()
            if stage == "preflight": preflight()
            elif stage == "noise": noise()
            elif stage == "round1":
                if not (ROOT / "analysis/noise_reference.json").exists(): raise ValueError("Complete experiment 1 first")
                # Prepare the already-required H0 measurements before any
                # proposal, then test actual diagnostic rows with the native
                # CLI. round_run reuses these stages without additional calls.
                def prepare_start(run):
                    create_run(run)
                    run_stage(run, BASE, 0, "score")
                    run_stage(run, BASE, 0, "feedback")
                two_at_a_time(prepare_start, CORE_RUNS)
                from pilot.verify_cli import run_check
                if not run_check()["passed"]:
                    raise RuntimeError("Actual-data CLI reading acceptance failed; paid generation is disabled")
                two_at_a_time(lambda run: round_run(run, 1), CORE_RUNS)
            elif stage == "round2":
                if not all((control_for(run) / "round1_checkpoint.json").exists() for run in CORE_RUNS): raise ValueError("Complete experiment 2 first")
                from pilot.verify_cli import run_check
                if not run_check()["passed"]:
                    raise RuntimeError("CLI acceptance failed before round two")
                two_at_a_time(lambda run: round_run(run, 2), CORE_RUNS)
            elif stage == "predict":
                from pilot.analyze import freeze_prediction
                freeze_prediction()
            elif stage == "holdout":
                from pilot.analyze import verify_prediction
                verify_prediction()
                round_run("D50_c", 1)
                round_run("D50_c", 2)
            elif stage == "analyze":
                from pilot.analyze import analyze_all
                analyze_all(include_holdout=4 in enabled)
        except Exception as exc:
            atomic_json(ROOT / "external/status.json", {"stage": stage, "state": "paused_on_error", "error_type": type(exc).__name__, "message": str(exc)[:1500], "updated_at_utc": utc_now()})
            raise
        atomic_json(ROOT / "external/status.json", {"stage": stage, "state": "complete", "updated_at_utc": utc_now()})


if __name__ == "__main__":
    with exclusive_controller(ROOT / "external/controller.lock"), exclusive_controller(ROOT / "external/trajectory_recovery.lock"):
        main()
