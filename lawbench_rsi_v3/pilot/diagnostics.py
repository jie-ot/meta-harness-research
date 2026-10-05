"""Lossless per-item reading views over immutable solver traces.

The compact JSONL carries complete inputs, predictions and answers. Long
prompts, responses and metadata stay available in separate detail files, so
one repeated 30k-character memory does not make the diagnostic row unreadable.
This performs no model calls and changes no scores, examples or memory state.
"""
from __future__ import annotations

import json
from pathlib import Path

from pilot.state import atomic_json, digest, jsonl_write, read_json, read_jsonl, utc_now


def materialize(directory: Path, phase: str):
    directory = Path(directory)
    trace_path = directory / f"{phase}_traces.jsonl"
    if not trace_path.exists():
        return None  # The original H0 training log has its own compact format.
    calls_path = directory / "calls.jsonl"
    source_hashes = {trace_path.name: digest(trace_path), calls_path.name: digest(calls_path)}
    marker = directory / "diagnostics_manifest.json"
    if marker.exists():
        saved = read_json(marker)
        if saved["source_hashes"] != source_hashes:
            raise ValueError("Immutable diagnostic source changed")
        for rel, expected in saved["files"].items():
            if digest(directory / rel) != expected:
                raise ValueError("Diagnostic view was changed")
        return saved
    traces = read_jsonl(trace_path)
    returned = {}
    for call in read_jsonl(calls_path):
        if call["event"] in {"api_result", "cache_hit"}:
            returned[call["call_id"]] = call
    rows, files = [], []
    for line, trace in enumerate(traces, 1):
        detail = directory / "details" / trace["item_id"]
        detail.mkdir(parents=True, exist_ok=True)
        atomic_json(detail / "metadata.json", trace.get("metadata", {}))
        files.append(detail / "metadata.json")
        references = []
        for index, call_id in enumerate(trace["call_ids"], 1):
            call = returned.get(call_id)
            if call is None:
                continue
            if call["item_id"] != trace["item_id"]:
                raise ValueError("Mismatched item and actual model call")
            names = {"prompt": f"call_{index}.prompt.txt", "response": f"call_{index}.response.txt", "request_metadata": f"call_{index}.json"}
            (detail / names["prompt"]).write_text(call.get("prompt", ""), encoding="utf-8")
            (detail / names["response"]).write_text(call.get("content", ""), encoding="utf-8")
            atomic_json(detail / names["request_metadata"], {"call_id":call_id,"model":call.get("model"),"system_prompt":call.get("system_prompt"),
                                                        "parameters":call.get("parameters"),"raw_response":call.get("raw_response"),
                                                        "raw_response_missing":call.get("raw_response") is None})
            files.extend(detail / name for name in names.values())
            references.append({key:(detail / name).relative_to(directory).as_posix() for key,name in names.items()})
        compact = {key:trace[key] for key in ["run_id","candidate","round","phase","D","item_id","input","prediction","target","was_correct","legacy"] if key in trace}
        compact.update({"metadata_file":(detail / "metadata.json").relative_to(directory).as_posix(),"model_calls":references,
                        "original_trace_file":trace_path.name,"original_trace_line":line})
        rows.append(compact)
    jsonl_write(directory / "diagnostics.jsonl", rows)
    files.append(directory / "diagnostics.jsonl")
    report = {"created_at_utc":utc_now(),"source_hashes":source_hashes,"count":len(rows),"input_prediction_target_unabridged":True,
              "files":{p.relative_to(directory).as_posix():digest(p) for p in files},
              "max_row_characters":max((len(json.dumps(r,ensure_ascii=False)) for r in rows),default=0)}
    atomic_json(marker,report)
    return report
