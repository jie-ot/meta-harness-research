"""Account for setup failures and observed feedback access after core analysis.

Reads saved artifacts only. It neither chooses harnesses nor calls a model.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pilot.state import Ledger, atomic_json, digest, read_json, read_jsonl, utc_now

RUNS = ["D0_a", "D0_b", "D100_a", "D100_b"]
OUT = ROOT / "analysis"


def write_csv(name, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def feedback_access():
    rows = []
    for run in RUNS:
        for t in [1, 2]:
            folder = ROOT / "external/control" / run / f"proposer_round{t}"
            events_path = next((folder / "sessions").glob("*/events.jsonl"))
            events = read_jsonl(events_path)
            tools = {block["id"]:block for event in events if event.get("type") == "assistant"
                     for block in event.get("message", {}).get("content", []) if block.get("type") == "tool_use"}
            visible, complete = defaultdict(set), defaultdict(set)
            attempted, errors = defaultdict(int), defaultdict(int)
            for event in events:
                if event.get("type") != "user":
                    continue
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    tool = tools.get(block["tool_use_id"], {})
                    args = tool.get("input", {})
                    path = (args.get("file_path") or args.get("path") or "").replace("\\", "/")
                    match = re.search(r"/history/([^/]+)/(score|feedback)/", "/" + path)
                    if not match or tool.get("name") not in {"Read", "Grep"}:
                        continue
                    candidate, phase = match.groups()
                    key = (candidate, phase)
                    attempted[key] += 1
                    if block.get("is_error"):
                        errors[key] += 1
                        continue
                    if tool["name"] == "Grep" and args.get("output_mode") != "content":
                        continue
                    content = str(block.get("content", ""))
                    visible[key].update(re.findall(r"lawbench_3-3_\d{4}", content))
                    if not path.endswith("diagnostics.jsonl"):
                        continue
                    for line in content.splitlines():
                        brace = line.find("{")
                        if brace < 0:
                            continue
                        try:
                            item = json.loads(line[brace:])
                        except (ValueError, TypeError):
                            continue
                        if isinstance(item, dict) and all(k in item for k in ["item_id", "input", "prediction", "target", "was_correct"]):
                            complete[key].add(item["item_id"])
            package = ROOT / "runs" / run / "text_classification"
            for diagnostics in (package / "history").glob("*/*/diagnostics.jsonl"):
                candidate, phase = diagnostics.parent.parent.name, diagnostics.parent.name
                if phase not in {"score", "feedback"}:
                    continue
                # Include only evidence available at this proposal's start.
                signature = read_json(diagnostics.parent / "start.json")["signature"]
                if signature["round"] >= t:
                    continue
                key = (candidate, phase)
                available = len(read_jsonl(diagnostics))
                rows.append({"run_id":run,"T":t,"D":read_json(ROOT / "pilot_config.json")["runs"][run],
                             "source_candidate":candidate,"phase":phase,"available_items":available,
                             "attempted_reads":attempted[key],"failed_reads":errors[key],
                             "visible_item_ids":len(visible[key]),"complete_diagnostic_rows_returned":len(complete[key]),
                             "note":"Returned text measures visibility; it does not prove how the optimizer used each item."})
    return rows


def main():
    analysis = read_json(OUT / "analysis.json")
    assert analysis["completed_experiments"] == [1,2,3]
    with (OUT / "checkpoint_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        summary = {(r["run_id"],int(r["T"])):r for r in csv.DictReader(handle)}
    ledger = [r for r in read_jsonl(ROOT / "external/cost_ledger.jsonl")
              if not r.get("run_id", "").startswith("D100_injected_")]
    indexed = {event["event_id"]:event for event in ledger}
    assert len(indexed) == len(ledger)
    failed_ids = set()
    for run in RUNS:
        recovery = ROOT / "external/control" / run / "recovery.json"
        if recovery.exists():
            failed_ids.update(read_json(recovery)["failed_setup_event_ids"])
    details = []
    for run in RUNS:
        for t in [1,2]:
            checkpoint = read_json(ROOT / "external/control" / run / f"round{t}_checkpoint.json")
            events = [indexed[key] for key in checkpoint["cost"]["ledger_event_ids"]]
            paid = [event for event in events if event["event"] in {"api_result","proposer_result"}]
            gross = sum(event.get("cost") or 0 for event in paid)
            setup = sum(event.get("cost") or 0 for event in paid if event["event_id"] in failed_ids)
            assert abs(gross - checkpoint["cost"]["cost_usd"]) < 1e-8
            closed = {e["attempt_id"] for e in events if e["event"] in {"api_result","proposer_result"} or e["event"] == "api_failed" and e.get("http_status") == 402}
            unknown = [e for e in events if e["event"] in {"api_started","proposer_started"} and e["attempt_id"] not in closed]
            stage_cost = {phase:sum(e.get("cost") or 0 for e in paid if e["phase"] == phase and e["event_id"] not in failed_ids)
                          for phase in ["proposer","train","score","feedback"]}
            assert abs(sum(stage_cost.values()) - (gross-setup)) < 1e-8
            row = summary[(run,t)]
            details.append({"run_id":run,"T":t,"N":checkpoint["N"],"D":checkpoint["D"],"selected":checkpoint["selected"],"S_pp":float(row["S_pp"]),
                            "gross_C_usd":gross,"setup_failure_cost_usd":setup,"experimental_C_usd":gross-setup,
                            "gross_unknown_reserve_usd":sum(e["reserved_usd"] for e in unknown),
                            "experimental_unknown_reserve_usd":sum(e["reserved_usd"] for e in unknown if e["event_id"] not in failed_ids),
                            "valid_new_candidates_so_far":sum(r["status"] == "valid" for r in checkpoint["candidates"][1:]),
                            **{f"{phase}_usd":value for phase,value in stage_cost.items()}})
    categories = defaultdict(float)
    for event in ledger:
        if event["event"] not in {"api_result","proposer_result"}:
            continue
        category = "setup_failure" if event["event_id"] in failed_ids else "frozen_measurement" if event["run_id"].startswith("noise_") else "external_test" if event["phase"] == "audit" else "experiment_search"
        categories[category] += event.get("cost") or 0
    budget = Ledger(ROOT / "external/cost_ledger.jsonl",80)._summary(ledger)
    assert abs(sum(categories.values()) - budget["known_or_estimated_usd"]) < 1e-8
    valid_search = sum(row["experimental_C_usd"] for row in details if row["T"] == 2)
    assert abs(valid_search - categories["experiment_search"]) < 1e-8
    invalid = [read_json(path) for path in (ROOT / "external/control").glob("*/invalid/*.json")]
    setup_candidates = []
    archive_path = ROOT / "external/运行过程归档.zip"
    archive = zipfile.ZipFile(archive_path) if archive_path.exists() else None
    try:
        for failure in ["cli_setup_20261003", "trace_readability_20261003"]:
            for run in RUNS:
                rel = f"external/failures/{failure}/runs/{run}/text_classification/pending_eval.json"
                if (ROOT / rel).exists():
                    pending = read_json(ROOT / rel)
                elif archive and rel in archive.namelist():
                    pending = json.loads(archive.read(rel).decode("utf-8"))
                else:
                    continue
                setup_candidates.extend({"failure": failure, "run_id": run, "candidate": candidate["name"], "source": rel}
                                        for candidate in pending["candidates"])
    finally:
        if archive:
            archive.close()
    write_csv("checkpoint_cost_detail.csv",details)
    write_csv("optimizer_evidence_access.csv",feedback_access())
    record = {"created_at_utc":utc_now(),"ledger_sha256":digest(ROOT / "external/cost_ledger.jsonl"),"cost_categories_usd":dict(categories),"budget":budget,
              "planned_proposer_sessions":8,"actual_paid_proposer_attempts":sum(e["event"] == "proposer_started" for e in ledger),
              "archived_setup_proposer_attempts":sum(e["event"] == "proposer_started" and e["event_id"] in failed_ids for e in ledger),
              "archived_setup_generated_candidates":setup_candidates,
              "nominal_current_candidates":analysis["nominal_new_candidates"],"valid_current_candidates":analysis["valid_new_candidates"],"invalid_current_candidates":invalid,
              "nonbillable_credit_rejections":sum(e["event"] == "api_failed" and e.get("http_status") == 402 for e in ledger),
              "cost_interpretation":"Gross C retains all recorded run expenditure. Experimental C removes separately archived infrastructure attempts; invalid generated candidates remain charged. Unknown reservations are bounds used for budget control, not reported charges."}
    atomic_json(OUT / "accounting_review.json",record)
    print({"known_total_usd":budget["known_or_estimated_usd"],"unknown_reserve_usd":budget["reserved_or_unknown_usd"],"categories":dict(categories),"invalid_candidates":len(invalid)})


if __name__ == "__main__":
    main()
