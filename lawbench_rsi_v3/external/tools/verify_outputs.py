"""Independent read-only reconstruction of saved experimental outputs.

Run with the original project Python; --final requires all planned artifacts.
The report writes hashes and counts, never credentials or case text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
from pilot.state import Ledger, atomic_json, digest, object_hash, read_json, read_jsonl, utc_now
from text_classification.data.evaluators import eval_lawbench


def verify(final=False):
    manifest = read_json(ROOT / "external/prepared_data_v3/manifest_v3.json")
    prepared = ROOT / "external/prepared_data_v3"
    panels = {}
    for name, split in manifest["splits"].items():
        assert digest(prepared / split["file"]) == split["sha256"]
        rows = read_jsonl(prepared / split["file"])
        assert [r["item_id"] for r in rows] == split["item_ids"]
        assert len(rows) == split["count"]
        panels[name] = rows
    source = ROOT.parent / "reference_examples/text_classification"
    original = read_json(ROOT / "preflight.json")["source_inventory_sha256"]
    for rel, expected in original.items():
        assert digest(source / rel) == expected, f"Original source changed: {rel}"
    stages = list((ROOT / "external/noise").glob("*/*/*/complete.json"))
    stages += list((ROOT / "external/audit").glob("*/*/complete.json"))
    stages += list((ROOT / "runs").glob("*/text_classification/history/*/*/complete.json"))
    # The original 1/2/3 cohort remains a sealed result. Supplementary runs,
    # including copied H0 views, are verified separately and never double-counted.
    stages = [p for p in stages if not any(part.startswith("D100_injected_") for part in p.parts)]
    accounting = []
    phase_counts = Counter()
    included_phase_counts = Counter()
    invalid_names = {(path.parent.parent.name, path.stem) for path in (ROOT / "external/control").glob("*/invalid/*.json")}
    for marker in stages:
        done = read_json(marker)
        for rel, expected in done["files"].items():
            assert digest(marker.parent / rel) == expected, f"Artifact changed: {marker.parent / rel}"
        signature = done["signature"]
        phase, run = signature["phase"], signature["run_id"]
        assert signature["manifest_hash"] == digest(prepared / "manifest_v3.json")
        traces = read_jsonl(marker.parent / f"{phase}_traces.jsonl")
        calls = read_jsonl(marker.parent / "calls.jsonl")
        result = read_json(marker.parent / "result.json")
        panel = panels[phase][:signature["D"]] if phase == "feedback" else panels[phase]
        assert [r["item_id"] for r in traces] == [r["item_id"] for r in panel], f"Item/order mismatch: {marker}"
        assert len(traces) == result["total"]
        assert sum(r["was_correct"] for r in traces) == result["correct"]
        assert abs(result["accuracy"] - (result["correct"] / result["total"] if result["total"] else 0)) < 1e-10
        indexed = {}
        for call in calls:
            indexed.setdefault(call["call_id"], []).append(call)
            if call["event"] in {"api_result", "cache_hit"}:
                assert call["model"] == "openrouter/openai/gpt-oss-120b"
                assert call["parameters"]["temperature"] == 0.0
                assert call["parameters"]["max_tokens"] == 16384
                assert call["system_prompt"] == "Reasoning: medium"
        for trace in traces:
            assert eval_lawbench(trace["prediction"], trace["target"])["correct"] == trace["was_correct"]
            for call_id in trace["call_ids"]:
                assert call_id in indexed
                assert all(c["item_id"] == trace["item_id"] and c["run_id"] == run and c["candidate"] == signature["candidate"] for c in indexed[call_id])
        if run.startswith("noise_"):
            assert result["cache_mode"] == "off" and result["usage"]["cache_hits"] == 0
        else:
            assert Path(signature["cache_dir"]).resolve() == (ROOT / "external/caches" / run).resolve()
        phase_counts[("noise_" if run.startswith("noise_") else "") + phase] += len(traces)
        if (run, signature["candidate"]) not in invalid_names:
            included_phase_counts[("noise_" if run.startswith("noise_") else "") + phase] += len(traces)
        accounting.append({"path": marker.relative_to(ROOT).as_posix(), "sha256": digest(marker), "run_id": run,
                           "phase": phase, "candidate": signature["candidate"], "items": len(traces), "recomputed_correct": result["correct"]})
    checkpoints = []
    session_count = 0
    nominal_candidates = []
    ledger = read_jsonl(ROOT / "external/cost_ledger.jsonl")
    assert len({r["event_id"] for r in ledger}) == len(ledger)
    by_id = {r["event_id"]: r for r in ledger}
    for run in ["D0_a", "D0_b", "D100_a", "D100_b"]:
        control = ROOT / "external/control" / run
        initialized = control / "initialized.json"
        if not initialized.exists():
            assert not final, f"Run missing: {run}"
            continue
        package = ROOT / "runs" / run / "text_classification"
        for rel, expected in read_json(initialized)["fingerprint"]["framework"].items():
            assert digest(package / rel) == expected
        for t in [1, 2]:
            folder = control / f"proposer_round{t}"
            marker = folder / "complete.json"
            if marker.exists():
                session_count += 1
                for rel, expected in read_json(marker)["files"].items():
                    assert digest(folder / rel) == expected
                assert read_json(folder / "execution_acceptance.json")["passed"]
                pending = read_json(folder / "pending_eval.json")
                assert len(pending["candidates"]) == 2
                for candidate in pending["candidates"]:
                    assert digest(package / candidate["file"]) == candidate["code_hash"]
                    nominal_candidates.append((run,t,candidate["name"]))
            else:
                assert not final, f"Proposer incomplete: {run}/{t}"
            cp = control / f"round{t}_checkpoint.json"
            if not cp.exists():
                assert not final, f"Checkpoint missing: {run}/{t}"
                continue
            saved = read_json(cp)
            for rel, expected in saved["artifacts"].items():
                assert digest(ROOT / rel) == expected
            best = max([r for r in saved["candidates"] if r["status"] == "valid"], key=lambda r:r["correct"])
            assert best["candidate"] == saved["selected"]
            assert saved["N"] == 2*t
            events = [by_id[id] for id in saved["cost"]["ledger_event_ids"]]
            assert object_hash(events) == saved["cost"]["cost_events_hash"]
            expected_cost = sum(r.get("cost") or 0 for r in events if r["event"] in {"api_result", "proposer_result"})
            assert abs(expected_cost - saved["cost"]["cost_usd"]) < 1e-9
            if t == 1:
                assert not any(r["phase"] == "feedback" and r["candidate"] != "confusion_disambiguation_memory" for r in events)
            checkpoints.append({"run":run,"T":t,"selected":saved["selected"],"gross_C_usd":expected_cost})
    if final:
        assert session_count == 8 and len(checkpoints) == 8
        assert len(nominal_candidates) == 16
        valid = [candidate for candidate in nominal_candidates if (candidate[0],candidate[2]) not in invalid_names]
        feedback_items = sum(read_json(ROOT / "pilot_config.json")["runs"][run] * (1 + sum(r==run and t==1 for r,t,name in valid)) for run in ["D0_a","D0_b","D100_a","D100_b"])
        expected = {"noise_score":600,"noise_audit":600,"train":200*len(valid),"score":400+100*len(valid),"feedback":feedback_items,"audit":100*len(valid)}
        assert included_phase_counts == expected, (included_phase_counts,expected)
        assert not (ROOT / "runs/D50_c").exists()
    report = {"verified_at_utc":utc_now(),"status":"PASS" if final else "PASS_FOR_CURRENT_COMPLETED_ARTIFACTS",
              "final":final,"model_calls":0,"original_files_unchanged":len(original),"completed_stages":len(accounting),
              "phase_item_counts":dict(phase_counts),"included_phase_item_counts":dict(included_phase_counts),
              "invalid_candidates":[{"run":r,"candidate":n} for r,n in sorted(invalid_names)],
              "accepted_proposer_sessions":session_count,"checkpoints":checkpoints,
              "budget":Ledger(ROOT / "external/cost_ledger.jsonl",80)._summary(
                  [r for r in ledger if not r.get("run_id", "").startswith("D100_injected_")]),"artifacts":accounting}
    atomic_json(ROOT / "analysis/output_verification.json",report)
    return report


def verify_injected():
    """Recompute the four-candidate supplement without any model calls."""
    import hashlib
    from statistics import mean
    from pilot.state import verify_done
    source_inventory = read_json(ROOT / "preflight.json")["source_inventory_sha256"]
    for rel, expected in source_inventory.items():
        assert digest(ROOT.parent / "reference_examples/text_classification" / rel) == expected
    runs = ["D100_injected_a", "D100_injected_b"]
    baseline = "confusion_disambiguation_memory"
    cfg = read_json(ROOT / "pilot_config.json")
    prepared = ROOT / "external/prepared_data_v3"
    manifest = read_json(prepared / "manifest_v3.json")
    manifest_hash = digest(prepared / "manifest_v3.json")
    pack = ROOT / "external/control/D100_injected_a/feedback_injected.jsonl"
    # The wrapper injects read_text() through UTF-8 stdin. On Windows this
    # normalizes file CRLF to LF; compare the request to those actual bytes.
    prompt_feedback_hash = hashlib.sha256(pack.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    records = read_jsonl(pack)
    header, feedback = records[0], records[1:]
    source = ROOT / "runs/D100_a/text_classification/history" / baseline / "feedback"
    originals = read_jsonl(source / "diagnostics.jsonl")
    assert len(feedback) == len(originals) == 100
    for row, original in zip(feedback, originals):
        assert header["shared_input_prefix"] + row["case_text"] == original["input"]
        assert all(row[k] == original[k] for k in ["item_id", "prediction", "target", "was_correct"])
        assert len(row["model_responses"]) == len(original["model_calls"])
        for response, ref in zip(row["model_responses"], original["model_calls"]):
            assert response["content"] == (source / ref["response"]).read_text(encoding="utf-8")
    for name, split in manifest["splits"].items():
        assert digest(prepared / split["file"]) == split["sha256"]
    budget = read_json(ROOT / "external/injected_budget.json")
    ledger = read_jsonl(ROOT / "external/cost_ledger.jsonl")
    events = [e for e in ledger if e.get("run_id") in runs]
    proposer_events = [e for e in events if e["event"] == "proposer_result"]
    proposer_usd = sum(e["cost"] for e in proposer_events)
    assert all("cost_usd" in r for r in budget["requests"].values()), "Unknown proposer cost remains reserved"
    assert abs(proposer_usd - sum(r["cost_usd"] for r in budget["requests"].values())) < 1e-7
    assert cfg["injected"]["proposer_budget_usd"] == 30
    assert cfg["injected"]["proposer_budget_mode"] == "notify_and_ask_no_immediate_termination"
    rows, groups, artifacts, phase_counts, source_costs, d0_rows = [], [], {}, Counter(), [], []
    sealed = []
    for run in runs:
        control = ROOT / "external/control" / run
        proposer = control / "proposer_round1"
        marker = read_json(proposer / "complete.json")
        verify_done(proposer, marker["signature"])
        execution = read_json(proposer / "execution_acceptance.json")
        for rel, sha in (execution.get("reused_native_execution") or {}).get("artifacts", {}).items():
            assert digest(ROOT / rel) == sha
        sealed.append(marker["completed_at_utc"])
    for run in runs:
        control = ROOT / "external/control" / run
        package = ROOT / "runs" / run / "text_classification"
        proposer = control / "proposer_round1"
        for rel, sha in read_json(control / "initialized.json")["fingerprint"]["framework"].items():
            assert digest(package / rel) == sha
        reuse = read_json(control / "reused_h0.json")
        for phase, info in reuse["stages"].items():
            original = ROOT / info["source"]
            assert digest(original / "complete.json") == info["source_complete_hash"]
            assert digest(package / "history" / baseline / phase / "result.json") == info["result_hash"]
            source_costs.append({"run_id": run, "phase": phase, "source": info["source"],
                                 "historical_cost_usd": read_json(original / "result.json")["estimated_cost_usd"], "new_calls": 0})
        http_ids = {r["request_id"] for r in read_jsonl(proposer / "http_requests.jsonl")
                    if r.get("event") == "http_started" and r.get("path") == "/v1/messages"}
        assert http_ids
        for request_id in http_ids:
            request = budget["requests"][request_id]
            assert request["feedback_sha256"] == prompt_feedback_hash
            if request.get("initial_request") or "initial_request" not in request:
                assert request["complete_feedback_verified"]
        assert any(budget["requests"][rid]["complete_feedback_verified"] for rid in http_ids)
        checkpoint = read_json(control / "round1_checkpoint.json")
        for rel, sha in checkpoint["artifacts"].items():
            assert digest(ROOT / rel) == sha
            artifacts[rel] = sha
        selected = max(checkpoint["candidates"], key=lambda r: r["correct"] if r["correct"] is not None else -1)
        assert selected["candidate"] == checkpoint["selected"]
        pending = read_json(proposer / "pending_eval.json")["candidates"]
        assert len(pending) == 2
        for candidate in pending:
            name = candidate["name"]
            assert digest(package / candidate["file"]) == candidate["code_hash"]
            result_by_phase = {}
            for phase in ["train", "score", "audit"]:
                folder = ROOT / "external/audit" / run / name if phase == "audit" else package / "history" / name / phase
                done = read_json(folder / "complete.json")
                signature = done["signature"]
                verify_done(folder, signature)
                assert signature["run_id"] == run and signature["candidate"] == name and signature["round"] == 1
                assert signature["manifest_hash"] == manifest_hash and signature["code_hash"] == candidate["code_hash"]
                assert signature["cache_mode"] == "run" and Path(signature["cache_dir"]).resolve() == (ROOT / "external/caches" / run).resolve()
                assert signature["model_config"] == {k: cfg[k] for k in signature["model_config"]}
                if phase == "train":
                    assert signature["memory_hash"] is None
                else:
                    assert signature["memory_hash"] == digest(package / "history" / name / "train/memory.json")
                start = read_json(folder / "start.json")["started_at_utc"]
                assert start > max(sealed), "Evaluation started before both proposals were frozen"
                if phase == "audit":
                    assert start > checkpoint["created_at_utc"]
                traces = read_jsonl(folder / (phase + "_traces.jsonl"))
                result = read_json(folder / "result.json")
                assert [r["item_id"] for r in traces] == manifest["splits"][phase]["item_ids"]
                assert len(traces) == result["total"] == (200 if phase == "train" else 100)
                correct = sum(bool(eval_lawbench(t["prediction"], t["target"])["correct"]) for t in traces)
                assert correct == result["correct"] == sum(t["was_correct"] for t in traces)
                assert abs(result["accuracy"] - correct / len(traces)) < 1e-10
                actual_calls, actual_contents = {}, {}
                with (folder / "calls.jsonl").open(encoding="utf-8") as handle:
                    for line in handle:
                        call = __import__("json").loads(line)
                        if call["event"] not in {"api_result", "cache_hit"}:
                            continue
                        assert call["model"] == cfg["solver"]
                        assert call["parameters"]["temperature"] == cfg["temperature"]
                        assert call["parameters"]["max_tokens"] == cfg["max_tokens"]
                        assert call["system_prompt"] == cfg["system_prompt"]
                        assert call["run_id"] == run and call["candidate"] == name and call["phase"] == phase
                        actual_calls[call["call_id"]] = call["item_id"]
                        actual_contents[call["call_id"]] = call["content"]
                for trace in traces:
                    assert all(actual_calls[call_id] == trace["item_id"] for call_id in trace["call_ids"])
                phase_counts[phase] += len(traces)
                result_by_phase[phase] = result
                artifacts[(folder / "complete.json").relative_to(ROOT).as_posix()] = digest(folder / "complete.json")
            rows.append({"run_id": run, "slot": candidate["slot"], "candidate": name,
                         "score_pp": result_by_phase["score"]["correct"], "audit_pp": result_by_phase["audit"]["correct"],
                         "selected": name == checkpoint["selected"], "code_hash": candidate["code_hash"],
                         "solver_cost_usd": sum(e.get("cost") or 0 for e in events if e.get("candidate") == name and e.get("run_id") == run and e["event"] == "api_result"),
                         "solver_api_requests": sum(e.get("candidate") == name and e.get("run_id") == run and e["event"] == "api_started" for e in events)})
            # Diagnose output transformations using the already saved responses.
            # This is a retrospective comparison, not a fresh ablation run.
            from text_classification.memory_system import extract_json_field
            changes = Counter()
            for trace in traces:  # The final phase above is audit.
                meta = trace.get("metadata", {})
                if "union" in meta:
                    before = "[罪名]" + ";".join(meta["union"]) + "<eoa>" if meta["union"] else ""
                elif "raw_answer" in meta:
                    before = meta["raw_answer"]
                elif not meta.get("retried") and len(trace["call_ids"]) == 1:
                    before = extract_json_field(meta["full_response"], "final_answer")
                else:
                    continue
                was = bool(eval_lawbench(before or "", trace["target"])["correct"])
                now = bool(trace["was_correct"])
                changes.update(covered=1, before_correct=int(was), after_correct=int(now),
                               helped=int(not was and now), harmed=int(was and not now))
            rows[-1]["saved_response_postprocessing"] = {**dict(changes),
                "scope": "Before/after output transformation on the same saved audit responses; retry cases without a separate raw answer are excluded. Earlier training and prompts are held as observed, so this is not a controlled ablation."}
            if name == "act_decomposition_union_memory":
                import re
                from text_classification.llm import parse_harmony_response
                diagnostic = Counter()
                for trace in traces:
                    meta = trace["metadata"]
                    diagnostic.update(acts=meta["num_acts"], fallback_calls=meta["retries"],
                                      empty_predictions=int(not trace["prediction"]))
                    if meta["retries"] == meta["num_acts"] and len(trace["call_ids"]) == 2 * meta["num_acts"]:
                        for call_id in trace["call_ids"][::2]:
                            response = parse_harmony_response(actual_contents[call_id])
                            diagnostic.update(first_calls_examined=1, nonempty_first_responses=int(bool(response.strip())),
                                              first_responses_without_final_answer=int(not extract_json_field(response, "final_answer")))
                    def has_suffix(answer):
                        payload = re.sub(r"\[罪名\]|<eoa>", "", answer or "")
                        return any(part.strip().endswith("罪") for part in re.split(r"[;；]", payload) if part.strip())
                    diagnostic.update(predictions_with_terminal_crime_suffix=int(has_suffix(trace["prediction"])),
                                      targets_with_terminal_crime_suffix=int(has_suffix(trace["target"])))
                rows[-1]["saved_response_format_diagnostic"] = dict(diagnostic)
        matching = "D0_" + run[-1]
        old_checkpoint = ROOT / "external/control" / matching / "round1_checkpoint.json"
        old = read_json(old_checkpoint)
        for rel, sha in old["artifacts"].items():
            assert digest(ROOT / rel) == sha
        for child in old["candidates"]:
            if child["candidate"] == baseline:
                continue
            old_folder = ROOT / "external/audit" / matching / child["candidate"]
            old_marker = read_json(old_folder / "complete.json")
            verify_done(old_folder, old_marker["signature"])
            d0_rows.append({"run_id": matching, "candidate": child["candidate"], "score_pp": child["correct"],
                           "audit_pp": read_json(old_folder / "result.json")["correct"],
                           "selected": child["candidate"] == old["selected"]})
            artifacts[(old_folder / "complete.json").relative_to(ROOT).as_posix()] = digest(old_folder / "complete.json")
        artifacts[old_checkpoint.relative_to(ROOT).as_posix()] = digest(old_checkpoint)
        old_audit = read_json(ROOT / "external/audit" / matching / old["selected"] / "result.json")["correct"]
        if checkpoint["selected"] == baseline:
            noise_path = ROOT / "analysis/noise_reference.json"
            noise = read_json(noise_path)
            h0_audits = []
            for rep in [1, 2]:
                folder = ROOT / f"external/noise/R4B/{rep}/audit"
                result_path = folder / "result.json"
                assert digest(result_path) == noise["input_hashes"][result_path.relative_to(ROOT).as_posix()]
                marker = read_json(folder / "complete.json")
                verify_done(folder, marker["signature"])
                assert marker["signature"]["code_hash"] == digest(package / "agents" / (baseline + ".py"))
                assert marker["signature"]["memory_hash"] == digest(package / "history" / baseline / "train/memory.json")
                h0_audits.append(read_json(result_path)["correct"])
                artifacts[(folder / "complete.json").relative_to(ROOT).as_posix()] = digest(folder / "complete.json")
            selected_audit = mean(h0_audits)
            assert selected_audit == noise["S0_pp"]
            artifacts[noise_path.relative_to(ROOT).as_posix()] = digest(noise_path)
        else:
            selected_audit = next(r["audit_pp"] for r in rows if r["run_id"] == run and r["selected"])
        groups.append({"run_id": run, "matched_D0": matching, "selected": checkpoint["selected"], "selected_score_pp": selected["correct"],
                       "selected_audit_pp": selected_audit, "D0_selected_audit_pp": old_audit, "delta_vs_D0_pp": selected_audit-old_audit,
                       "selected_audit_source": "reused_H0_frozen_repeat_mean" if checkpoint["selected"] == baseline else "new_audit100",
                       "offspring_mean_audit_pp": mean(r["audit_pp"] for r in rows if r["run_id"] == run),
                       "new_algorithm_cost_usd_including_failures": checkpoint["cost"]["cost_usd"]})
    assert phase_counts == {"train":800, "score":400, "audit":400}
    assert not any(e.get("candidate") == baseline and e["event"] == "api_started" for e in events)
    assert not any(e.get("phase") == "feedback" and e["event"] == "api_started" for e in events)
    solver_usd = sum(e.get("cost",0) for e in events if e["event"] == "api_result")
    original_d100 = []
    for run in ["D100_a", "D100_b"]:
        old_path = ROOT / "external/control" / run / "round1_checkpoint.json"
        old = read_json(old_path)
        for rel, sha in old["artifacts"].items():
            assert digest(ROOT / rel) == sha
        children = []
        for child in old["candidates"]:
            if child["candidate"] == baseline:
                continue
            folder = ROOT / "external/audit" / run / child["candidate"]
            marker = read_json(folder / "complete.json")
            verify_done(folder, marker["signature"])
            children.append({"candidate": child["candidate"], "score_pp": child["correct"],
                             "audit_pp": read_json(folder / "result.json")["correct"], "selected": child["candidate"] == old["selected"]})
            artifacts[(folder / "complete.json").relative_to(ROOT).as_posix()] = digest(folder / "complete.json")
        original_d100.append({"run_id": run, "candidates": children,
            "selected_audit_pp": next(c["audit_pp"] for c in children if c["selected"]),
            "offspring_mean_audit_pp": mean(c["audit_pp"] for c in children)})
        artifacts[old_path.relative_to(ROOT).as_posix()] = digest(old_path)
    from pilot.state import Ledger
    supplement_accounting = Ledger(ROOT / "external/cost_ledger.jsonl", cfg["budget_usd"])._summary(events)
    report = {"status":"PASS", "verified_at_utc":utc_now(), "verification_model_calls":0,
              "original_files_unchanged":len(source_inventory),
              "feedback_records_inline":100, "feedback_sha256":digest(pack),
              "feedback_prompt_utf8_sha256":prompt_feedback_hash, "formal_candidates":len(rows),
              "independent_proposer_arms":2, "completed_final_proposer_sessions":2,
              "paid_proposer_attempts":len(proposer_events), "phase_item_counts":dict(phase_counts),
              "new_prediction_tasks":sum(phase_counts.values()), "proposer_total_usd_including_failures":proposer_usd,
              "unsuccessful_cli_sessions_usd":sum(e["cost"] for e in proposer_events if not e.get("success")),
              "solver_total_usd":solver_usd, "supplement_total_usd":proposer_usd+solver_usd,
              "supplement_reserved_or_unknown_usd":supplement_accounting["reserved_or_unknown_usd"],
              "selected_mean_audit_pp":mean(g["selected_audit_pp"] for g in groups),
              "selected_mean_delta_vs_D0_pp":mean(g["delta_vs_D0_pp"] for g in groups),
              "offspring_mean_audit_pp":mean(r["audit_pp"] for r in rows),
              "D0_offspring_mean_audit_pp":mean(r["audit_pp"] for r in d0_rows),
              "offspring_mean_delta_vs_D0_pp":mean(r["audit_pp"] for r in rows)-mean(r["audit_pp"] for r in d0_rows),
              "D0_candidates":d0_rows,
              "original_D100":original_d100,
              "original_D100_selected_mean_audit_pp":mean(r["selected_audit_pp"] for r in original_d100),
              "original_D100_offspring_mean_audit_pp":mean(r["offspring_mean_audit_pp"] for r in original_d100),
              "training_candidate_concurrency":cfg["injected"].get("training_candidate_concurrency",2),
              "candidates":rows,"comparisons":groups,"reused_H0_measurements":source_costs,"artifacts":artifacts,
              "scope":"Historical D0 comparison with two independent proposer arms and retained resumptions; audit is not used for selection. Copied H0 measurements are not new calls. User-authorized candidate parallelism preserves each sequential online training stream."}
    atomic_json(ROOT / "analysis/injected_results.json", report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--final",action="store_true")
    parser.add_argument("--injected",action="store_true")
    args=parser.parse_args()
    report=verify_injected() if args.injected else verify(args.final)
    print({k:v for k,v in report.items() if k not in {"artifacts","checkpoints"}})
