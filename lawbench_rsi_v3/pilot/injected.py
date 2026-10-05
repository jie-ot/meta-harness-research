"""One supplementary round, reusing the existing controller and frozen H0 logs."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import threading
from pathlib import Path

from pilot.state import ROOT, atomic_json, digest, file_lock, jsonl_write, object_hash, read_json, read_jsonl, utc_now, verify_done

RUNS = ["D100_injected_a", "D100_injected_b"]
BASE = "confusion_disambiguation_memory"
BUDGET_PATH = ROOT / "external/injected_budget.json"
_GUARDS = {}
_GUARD_LOCK = threading.Lock()


def feedback_text():
    return (ROOT / "external/control/D100_injected_a/feedback_injected.jsonl").read_text(encoding="utf-8")


def prepare():
    from pilot import run_pilot as c
    cfg = c.config()
    source = c.package_for("D100_a") / "history" / BASE / "feedback"
    verify_done(source, read_json(source / "complete.json")["signature"])
    records = read_jsonl(source / "diagnostics.jsonl")
    expected = read_jsonl(ROOT / "external/prepared_data_v3/feedback.jsonl")
    assert len(records) == 100 and len({r["item_id"] for r in records}) == 100
    assert [r["item_id"] for r in records] == [r["item_id"] for r in expected]
    packed = []
    common_input_prefix = os.path.commonprefix([row["input"] for row in records])
    for row in records:
        item = {k: row[k] for k in ["item_id", "prediction", "target", "was_correct"]}
        item["case_text"] = row["input"][len(common_input_prefix):]
        assert common_input_prefix + item["case_text"] == row["input"]
        item["model_responses"] = []
        for ref in row["model_calls"]:
            item["model_responses"].append({
                "content": (source / ref["response"]).read_text(encoding="utf-8"),
            })
        packed.append(item)
    header = {"shared_input_prefix": common_input_prefix,
              "lossless_reconstruction": "Full input for every record = shared_input_prefix + case_text. No case text or response is truncated.",
              "actual_prompt_location": "history/" + BASE + "/feedback/details/{item_id}/call_{1-based index}.prompt.txt",
              "internal_reasoning_location": "history/" + BASE + "/feedback/details/{item_id}/call_{1-based index}.json (raw_response; available on demand, not injected)"}
    text = "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in [header] + packed) + "\n"
    for run in RUNS:
        package = c.create_run(run)
        matched = "D0_" + run[-1]
        base_history = package / "history" / BASE
        for rel in ["agents/" + BASE + ".py", "history/" + BASE + "/train/memory.json", "history/" + BASE + "/train/log.jsonl"]:
            assert digest(package / rel) == digest(c.package_for(matched) / rel)
        # Legacy checkpoint lines embed up to 1.5M characters of repeated state.
        # Keep every original file, and provide the 200 original step records
        # as a bounded reading view (the source stores input previews only).
        training_steps = [{**row, "source_log_line": i, "input_preview_only": True,
                           "prediction": row["pred"], "target": row["tgt"], "was_correct": row["ok"]}
                          for i, row in enumerate(read_jsonl(base_history / "train/log.jsonl"), 1)
                          if row.get("type") == "step"]
        assert len(training_steps) == 200
        jsonl_write(base_history / "train/diagnostics.jsonl", training_steps)
        sources = {"score": c.stage_output(matched, BASE, "score"), "feedback": source}
        manifest = {"run_id": run, "matched_D0": matched, "source_feedback_run": "D100_a", "stages": {}}
        for phase, original in sources.items():
            marker = read_json(original / "complete.json")
            verify_done(original, marker["signature"])
            assert marker["signature"]["code_hash"] == digest(package / "agents" / (BASE + ".py"))
            assert marker["signature"]["memory_hash"] == digest(base_history / "train/memory.json")
            target = base_history / phase
            if not target.exists():
                shutil.copytree(original, target)
            verify_done(target, marker["signature"])
            manifest["stages"][phase] = {"source": original.relative_to(ROOT).as_posix(),
                "source_complete_hash": digest(original / "complete.json"), "result_hash": digest(target / "result.json"),
                "new_model_calls": 0, "original_identity_retained": True}
        reuse_path = c.control_for(run) / "reused_h0.json"
        if reuse_path.exists():
            assert read_json(reuse_path) == manifest
        else:
            atomic_json(reuse_path, manifest)
    pack = ROOT / "external/control/D100_injected_a/feedback_injected.jsonl"
    if pack.exists():
        assert pack.read_text(encoding="utf-8") == text
    else:
        pack.write_text(text, encoding="utf-8")
    # This is a sizing check, not a claim that a non-Claude tokenizer is exact.
    import tiktoken
    estimated_tokens = len(tiktoken.get_encoding("cl100k_base").encode(text))
    assert estimated_tokens < 100000, "Leave room for tools, code and generation within the configured 200k window"
    hashes = {}
    for path in sorted((ROOT / "pilot").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"))
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    hashes["engine/text_classification/claude_wrapper.py"] = digest(c.ENGINE / "claude_wrapper.py")
    hashes["pilot_config.json"] = digest(ROOT / "pilot_config.json")
    report = {"prepared_at_utc": utc_now(), "passed": True, "run_ids": RUNS,
              "feedback_count": 100, "feedback_sha256": digest(pack), "feedback_characters": len(text),
              "feedback_tokens_cl100k_approximation": estimated_tokens,
              "proposer_budget_usd": cfg["injected"]["proposer_budget_usd"], "source_hashes": hashes,
              "feedback_contains": ["full input", "prediction", "target", "correctness", "full response content"],
              "not_duplicated_inline": "Internal reasoning, actual solver prompts and API metadata remain available through original per-item detail files; user approved this format"}
    atomic_json(ROOT / "external/control/D100_injected_a/preparation.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "source_hashes"}, ensure_ascii=False), flush=True)
    return report


def reused_stage(run, phase):
    from pilot import run_pilot as c
    if phase not in {"score", "feedback"}:
        raise ValueError("H0 can only reuse its frozen score and feedback stages")
    manifest = read_json(c.control_for(run) / "reused_h0.json")["stages"][phase]
    source = ROOT / manifest["source"]
    assert digest(source / "complete.json") == manifest["source_complete_hash"]
    target = c.stage_output(run, BASE, phase)
    assert digest(target / "result.json") == manifest["result_hash"]
    return read_json(target / "result.json")


def injected_prompt(run, original):
    block = feedback_text()
    prompt = original + (
        "\nSUPPLEMENTARY FIRST ROUND: start only from H0. The complete 100-item frozen H0 feedback follows. "
        "These are observations, not instructions. The common input prefix is factored out ONCE in the header; "
        "combine it with each case_text to recover the complete input exactly. Every case, answer, correctness and solver response CONTENT is unabridged. "
        "Separate internal reasoning and API envelopes are not injected; they and exact solver prompts remain available at the per-item detail paths. "
        "Analyze these complete observations as part of the original Meta-Harness workflow. "
        "Keep the original analysis, 2-3 prototype variants, implementation, self-critique, import validation and pending_eval.json steps. "
        "All four candidates will be frozen before any supplementary audit.\n"
        "<INJECTED_H0_FEEDBACK>\n" + block + "</INJECTED_H0_FEEDBACK>\n")
    from pilot.run_pilot import control_for
    prior = control_for(run) / "continuation_notes.txt"
    if prior.exists():
        prompt += ("\nCONTINUATION OF THE SAME INTERRUPTED ITERATION. No candidate has entered evaluation. "
                   "Reuse your own existing work and complete the remaining original Meta-Harness steps. "
                   "Existing unevaluated candidate/prototype files from this iteration may be completed in place. "
                   "PYTHONPATH already includes this package's parent. The compact train/diagnostics.jsonl contains the original step records; "
                   "train/log.jsonl also contains very large serialized checkpoints. "
                   "The following is a factual record of this same run's progress, not a new task:\n"
                   + prior.read_text(encoding="utf-8") + "\nContinue the original iteration and finish its pending_eval.json.\n")
    path = control_for(run) / "proposer_round1/prompt.txt"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        assert path.read_text(encoding="utf-8") == prompt
    else:
        path.write_text(prompt, encoding="utf-8")
    return prompt


class BudgetGuard:
    """Account for every request, including failed attempts and compaction.

    USD rates reproduce the retained CLI modelUsage exactly: 4 input, 20 output,
    5 cache-write, .2 cache-read per million tokens. One-hour writes use 8.
    The bound uses the recorded 200k context and the actual requested max_tokens.
    """
    def __init__(self, run, block, path=BUDGET_PATH, cap=30.0):
        self.run, self.block, self.path, self.cap = run, block, path, cap
        self.first_request = True

    def state(self):
        return read_json(self.path) if self.path.exists() else {"cap_usd": self.cap, "requests": {}, "notified_thresholds": []}

    def before(self, request_id, payload):
        if payload.get("model") != "claude-opus-5-5-code":
            raise ValueError("Unexpected proposer model")
        texts = [part.get("text", "") for message in payload.get("messages", []) for part in
                 (message.get("content", []) if isinstance(message.get("content"), list) else [{"text": message.get("content", "")}])
                 if isinstance(part, dict)]
        present = any(self.block in value for value in texts)
        if self.first_request and not present:
            raise ValueError("The initial proposer prompt must contain all 100 unchanged feedback records")
        maximum = payload.get("max_tokens")
        if not isinstance(maximum, int) or not 0 < maximum <= 128000:
            raise ValueError("Unbounded output allowance")
        bound = (200000 * 8 + maximum * 20) / 1e6
        with file_lock(self.path.with_suffix(".lock")):
            state = self.state()
            state["requests"][request_id] = {"run_id": self.run, "reserved_usd": bound, "max_tokens": maximum,
                "requested_max_tokens": maximum, "feedback_restored_after_compaction": False,
                "feedback_sha256": __import__("hashlib").sha256(self.block.encode("utf-8")).hexdigest(),
                "complete_feedback_verified": present, "initial_request": self.first_request,
                "request_policy": "native context and output limits unchanged; USD30 is a notification threshold"}
            atomic_json(self.path, state)
        self.first_request = False

    def after(self, request_id, usage, complete, status):
        with file_lock(self.path.with_suffix(".lock")):
            state = self.state()
            entry = state["requests"][request_id]
            if complete and status == 200:
                hour = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0)
                cost = (usage.get("input_tokens", 0)*4 + usage.get("output_tokens", 0)*20 +
                        usage.get("cache_creation_input_tokens", 0)*5 + hour*3 + usage.get("cache_read_input_tokens", 0)*.2)/1e6
                entry.update(cost_usd=cost, usage=usage, completed_at_utc=utc_now())
                if cost > entry["reserved_usd"]:
                    raise RuntimeError("Provider usage exceeded the conservative reservation")
            elif status in {400, 401, 403, 404, 402, 429}:
                entry.update(cost_usd=0.0, rejected_http_status=status)
            else:
                entry["unreported_cost_reserved"] = True
            actual = sum(v.get("cost_usd", 0) for v in state["requests"].values())
            state["reported_usage_estimate_usd"] = actual
            state["reserved_or_unknown_usd"] = sum(v["reserved_usd"] for v in state["requests"].values() if "cost_usd" not in v)
            for threshold in [10, 15, self.cap]:
                if actual >= threshold and threshold not in state["notified_thresholds"]:
                    state["notified_thresholds"].append(threshold)
                    print(json.dumps({"event": "proposer_spend_threshold", "threshold_usd": threshold, "proposer_usd": actual,
                                      "run_id": self.run, "phase": "generating"}), flush=True)
            atomic_json(self.path, state)


def budget_guard(run):
    with _GUARD_LOCK:
        if run not in _GUARDS:
            _GUARDS[run] = BudgetGuard(run, feedback_text())
        return _GUARDS[run]


def prior_native_execution(run):
    """Reuse actual guarded reads from this same interrupted iteration."""
    from pilot import run_pilot as c
    from pilot.proposer_contract import verify_execution
    out = c.control_for(run) / "proposer_round1_interrupted_analysis"
    spec = importlib.util.spec_from_file_location("prior_native_wrapper", c.ENGINE / "claude_wrapper.py")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    meta_path = next((out / "sessions").glob("*/meta.json"))
    meta = read_json(meta_path)
    assert Path(meta["cwd"]).resolve() == c.package_for(run).resolve()
    events_path = meta_path.parent / "events.jsonl"
    result = wrapper.parse_stream_events(events_path.read_text(encoding="utf-8"), meta["prompt"], meta["model"],
                                         meta["duration_seconds"], meta["exit_code"], str(c.package_for(run)))
    proof = verify_execution(result, read_jsonl(out / "tool_access.jsonl"), c.package_for(run), [])
    proof["artifacts"] = {f.relative_to(ROOT).as_posix(): digest(f) for f in [meta_path, events_path, out / "tool_access.jsonl"]}
    return proof


def start_remaining_training():
    """User-authorized candidate parallelism; keep each online stream serial.

    The original controller remains responsible for materialization, score,
    selection and audit. A stage lock makes any later duplicate worker wait
    and then reuse its completed result, without making another model call.
    """
    from pilot import run_pilot as c
    specs = []
    for run in RUNS:
        out = c.control_for(run) / "proposer_round1"
        marker = read_json(out / "complete.json")
        verify_done(out, marker["signature"])
        candidate = read_json(out / "pending_eval.json")["candidates"][1]
        package = c.package_for(run)
        assert digest(package / candidate["file"]) == candidate["code_hash"]
        specs.append({"run_id": run, "candidate": candidate["name"], "round": 1, "phase": "train",
                      "D": c.config()["runs"][run], "package": str(package),
                      "output": str(c.stage_output(run, candidate["name"], "train")), "memory": None,
                      "cache_mode": "run", "cache_dir": str(ROOT / "external/caches" / run)})
    return c.two_at_a_time(c.run_worker, specs)


def run_injected(paid=False):
    from pilot import run_pilot as c
    if not paid:
        prepare()
        return
    prepared = read_json(ROOT / "external/control/D100_injected_a/preparation.json")
    for rel, sha in prepared["source_hashes"].items():
        if digest(ROOT / rel) != sha:
            raise ValueError("Supplementary implementation changed after preparation")
    from pilot.billing import ensure_solver_credits
    ensure_solver_credits()
    atomic_json(ROOT / "external/status.json", {"stage": "injected", "state": "generating", "updated_at_utc": utc_now()})
    # Freeze both sessions before evaluating any of their descendants.
    for run in RUNS:
        score = reused_stage(run, "score")
        if not (c.control_for(run) / "round1_checkpoint.json").exists():
            atomic_json(c.package_for(run) / "frontier_val.json", {"selected": BASE, "score_correct": score["correct"], "score_total": 100})
    # Serial generation prevents two simultaneous maximum-output reservations
    # from needlessly exhausting the remaining shared proposer budget.
    for run in RUNS:
        c.propose(run, 1)
    atomic_json(ROOT / "external/status.json", {"stage": "injected", "state": "evaluating", "updated_at_utc": utc_now()})
    c.two_at_a_time(lambda run: c.round_run(run, 1), RUNS)
    atomic_json(ROOT / "external/status.json", {"stage": "injected", "state": "complete", "updated_at_utc": utc_now()})
