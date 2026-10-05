"""One trusted evaluation stage; data stays outside proposer workspaces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot.state import (ROOT, BudgetExceeded, Ledger, atomic_json, digest, finish,
                         jsonl_write, object_hash, read_json, read_jsonl, usage_from_calls,
                         utc_now, verify_done, serial_stage)


def execute(spec: dict, *, llm_override=None):
    cfg = read_json(ROOT / "pilot_config.json")
    if llm_override is None and not cfg["budget_confirmed"]:
        raise PermissionError("Taskbook requires a user-confirmed budget before paid execution")
    package = Path(spec["package"]).resolve()
    sys.path.insert(0, str(package.parent))
    from text_classification.data.api import load_pilot_split
    from text_classification.inner_loop import evaluate_memory, load_memory_system, make_result, run_inner_loop
    from text_classification.llm import LLM
    from text_classification.pilot_observation import capture_calls, set_event_callback

    manifest = (ROOT / cfg["manifest"]).resolve()
    phase = spec["phase"]
    examples, evaluator = load_pilot_split(manifest, phase, spec["D"] if phase == "feedback" else None)
    directory = Path(spec["output"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    calls_path = directory / "calls.jsonl"
    code_path = package / "agents" / (spec["candidate"] + ".py")
    memory_path = Path(spec["memory"]).resolve() if spec.get("memory") else None
    model_config = {k: cfg[k] for k in ["solver", "temperature", "max_tokens", "system_prompt", "training_mode", "training_batch_size", "training_seed", "evaluation_workers", "max_api_retries"]}
    signature = {
        "run_id": spec["run_id"], "candidate": spec["candidate"], "round": spec["round"],
        "D": spec["D"], "phase": phase, "manifest_hash": digest(manifest),
        "item_ids_hash": object_hash([x["item_id"] for x in examples]),
        "code_hash": digest(code_path), "memory_hash": digest(memory_path) if memory_path else None,
        "model_config": model_config, "cache_mode": spec["cache_mode"],
        "cache_dir": str(Path(spec["cache_dir"]).resolve()) if spec.get("cache_dir") else None,
        "engine_hashes": {name: digest(package / name) for name in ["llm.py", "inner_loop.py", "memory_system.py", "pilot_observation.py", "data/api.py", "data/loaders.py", "data/evaluators.py"]},
    }
    complete = verify_done(directory, signature)
    if complete:
        return read_json(directory / "result.json")
    start_path = directory / "start.json"
    if start_path.exists():
        if read_json(start_path)["signature"] != signature:
            raise ValueError("Partial stage inputs changed; refusing a mixed evaluation")
    else:
        atomic_json(start_path, {"signature": signature, "started_at_utc": utc_now()})
    if not calls_path.exists():
        calls_path.touch()
    previous_usage = usage_from_calls(calls_path)
    if previous_usage["incomplete_attempts"]:
        raise RuntimeError("Unfinished API attempts require accounting review before resuming")

    ledger = Ledger(ROOT / "external" / "cost_ledger.jsonl", cfg["budget_usd"])
    set_event_callback(ledger.observe_api if llm_override is None else None)
    if llm_override is None:
        from dotenv import dotenv_values
        credentials = dotenv_values((ROOT / cfg["credentials_file"]).resolve())
        key = credentials.get("OPENROUTER_API_KEY")
        if not key:
            raise PermissionError("OPENROUTER_API_KEY is missing")
        llm = LLM(model=cfg["solver"], api_key=key, temperature=cfg["temperature"],
                  max_tokens=cfg["max_tokens"], cache_mode=spec["cache_mode"], cache_dir=spec.get("cache_dir"))
    else:
        llm = llm_override
    memory = load_memory_system(f"agents/{spec['candidate']}.py", llm)
    metadata = {k: spec[k] for k in ["run_id", "candidate", "round", "phase", "D"]}
    clock_start = time.monotonic()
    attempts_path = directory / "attempt_times.jsonl"
    traces = []
    try:
        if phase == "train":
            checkpoint_path = directory / "training_checkpoint.json"
            next_step = 0
            if checkpoint_path.exists():
                checkpoint = read_json(checkpoint_path)
                if checkpoint["signature_hash"] != object_hash(signature):
                    raise ValueError("Training checkpoint signature changed")
                next_step, traces = checkpoint["next_step"], checkpoint["traces"]
                memory.set_state(checkpoint["state"])
            for index in range(next_step, len(examples)):
                ex = examples[index]
                context = {**metadata, "item_id": ex["item_id"]}
                with capture_calls(context, calls_path) as calls:
                    result = run_inner_loop(memory, [ex], evaluator, batch_size=1,
                                            max_workers=1, step_offset=index, mode="online")
                trace = {**context, **result["trajectory"][0], "call_ids": list(dict.fromkeys(c["call_id"] for c in calls)), "calls_file": calls_path.name}
                traces.append(trace)
                atomic_json(checkpoint_path, {"signature_hash": object_hash(signature), "next_step": index + 1, "state": memory.get_state(), "traces": traces})
                if (index + 1) % 10 == 0:
                    print(json.dumps({**metadata, "completed": index + 1, "total": len(examples)}, ensure_ascii=False), flush=True)
            (directory / "memory.json").write_text(memory.get_state(), encoding="utf-8")
            aggregate = make_result(traces)
            trace_file = "train_traces.jsonl"
            persistent_changed = None
        else:
            if memory_path is None:
                raise ValueError("Evaluation requires a saved memory")
            saved_state = memory_path.read_text(encoding="utf-8")
            memory.set_state(saved_state)
            state_before = memory.get_state()
            item_dir = directory / "items"
            item_dir.mkdir(exist_ok=True)
            existing = {}
            for path in item_dir.glob("*.json"):
                row = read_json(path)
                if row["signature_hash"] != object_hash(signature):
                    raise ValueError("Per-item checkpoint signature changed")
                existing[int(path.stem)] = row["prediction"]
            def save_prediction(index, result):
                atomic_json(item_dir / f"{index:04d}.json", {"signature_hash": object_hash(signature), "prediction": result})
            result = evaluate_memory(memory, examples, evaluator, max_workers=cfg["evaluation_workers"],
                                     observation=metadata, calls_path=calls_path,
                                     completed_predictions=existing, on_prediction=save_prediction)
            traces = result["predictions"]
            aggregate = make_result(traces)
            aggregate["memory_context_chars"] = result.get("avg_context_len", 0)
            persistent_changed = memory.get_state() != state_before
            if persistent_changed:
                # The learning interface is never called for evaluation. Record observable state effects.
                atomic_json(directory / "prediction_state_effect.json", {"before_hash": hashlib.sha256(state_before.encode()).hexdigest(), "after_hash": hashlib.sha256(memory.get_state().encode()).hexdigest()})
            trace_file = f"{phase}_traces.jsonl"
        elapsed = time.monotonic() - clock_start
        active_seconds = sum(row["wall_seconds"] for row in read_jsonl(attempts_path)) + elapsed
        first_start = datetime.fromisoformat(read_json(start_path)["started_at_utc"])
        usage = usage_from_calls(calls_path)
        if usage["incomplete_attempts"]:
            raise RuntimeError("Incomplete API accounting at stage completion")
        output = {
            **aggregate, **metadata, "dataset": "LawBench", "model": cfg["solver"],
            "seed": 42, "mode": "online", "num_epochs": None, "timestamp": utc_now(),
            "manifest_hash": signature["manifest_hash"], "code_hash": signature["code_hash"],
            "memory_hash": digest(directory / "memory.json") if phase == "train" else signature["memory_hash"],
            "model_config": model_config, "cache_mode": spec["cache_mode"],
            "runtime_seconds_this_attempt": elapsed,
            "runtime_seconds": active_seconds,
            "elapsed_seconds_including_resume_gaps": (datetime.now(timezone.utc) - first_start).total_seconds(),
            "llm_calls": usage["logical_calls"], "llm_input_tokens": usage["input_tokens"],
            "llm_output_tokens": usage["output_tokens"],
            "llm_total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "estimated_cost_usd": usage["cost_usd"], "usage": usage,
            "persistent_prediction_state_changed": persistent_changed,
        }
        jsonl_write(directory / trace_file, traces)
        atomic_json(directory / "result.json", output)
        files = ["result.json", trace_file, "calls.jsonl"]
        if phase == "train":
            files.append("memory.json")
        if phase == "score":
            atomic_json(directory / "val.json", output)
            files.append("val.json")
        if phase == "audit":
            atomic_json(directory / "test.json", output)
            files.append("test.json")
        finish(directory, signature, files, usage=usage)
        # Per-item and per-step checkpoints are redundant only after verified completion.
        verify_done(directory, signature)
        for intermediate in [directory / "items", directory / "training_checkpoint.json"]:
            if intermediate.exists():
                resolved = intermediate.resolve()
                if resolved.parent != directory or not resolved.is_relative_to(ROOT):
                    raise ValueError("Unsafe checkpoint cleanup path")
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
        return output
    finally:
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ended_at_utc": utc_now(), "wall_seconds": time.monotonic() - clock_start}) + "\n")
        if llm_override is None:
            llm.__exit__()
        set_event_callback(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    try:
        gate = serial_stage(Path(spec["output"]) / "worker.lock") if spec["run_id"].startswith("D100_injected_") else nullcontext()
        with gate:
            result = execute(spec)
        print(json.dumps({k: result.get(k) for k in ["run_id", "candidate", "phase", "correct", "total", "estimated_cost_usd"]}, ensure_ascii=False))
    except Exception as exc:
        # Provider exception strings can contain request details. Keep a safe error record.
        error = {"error_type": type(exc).__name__, "timestamp_utc": utc_now(),
                 "http_status": getattr(exc, "status_code", None),
                 "phase": spec["phase"], "run_id": spec["run_id"], "candidate": spec["candidate"]}
        atomic_json(Path(spec["output"]) / "failure.json", error)
        print(json.dumps(error), file=sys.stderr)
        raise SystemExit(30)


if __name__ == "__main__":
    main()
