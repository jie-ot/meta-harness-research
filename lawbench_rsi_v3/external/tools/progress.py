"""Read-only progress, without opening the live large training checkpoints."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import ctypes
import os
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pilot.state import Ledger, read_json


def memory_status():
    if os.name != "nt":
        return {}
    class Status(ctypes.Structure):
        _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ["total_physical","available_physical","total_commit","available_commit","total_virtual","available_virtual","extended"]]
    value=Status();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        return {}
    return {"free_physical_GiB":round(value.available_physical/1024**3,2),"free_commit_GiB":round(value.available_commit/1024**3,2)}


def logged_training_count(folder):
    result=0
    path=folder / "worker.log"
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try:
            row=json.loads(line)
        except ValueError:
            continue
        if isinstance(row,dict) and row.get("phase")=="train" and "completed" in row:
            result=max(result,row["completed"])
    return result


def progress():
    cfg=read_json(ROOT / "pilot_config.json")
    budget=Ledger(ROOT / "external/cost_ledger.jsonl",cfg["budget_usd"]).summary()
    pending=[]
    for run in cfg["runs"]:
        control=ROOT / "external/control" / run
        for t in [1,2]:
            proposer=control / f"proposer_round{t}"
            if not (proposer / "started.json").exists():
                continue
            if not (proposer / "complete.json").exists():
                access=proposer / "tool_access.jsonl"
                pending.append({"run":run,"T":t,"phase":"proposer","tool_events":len(access.read_text(encoding="utf-8").splitlines()) if access.exists() else 0})
                continue
            for candidate in read_json(proposer / "pending_eval.json")["candidates"]:
                name=candidate["name"]
                if (control / "invalid" / f"{name}.json").exists():
                    continue
                feedback=ROOT / "runs" / run / "text_classification/history" / name / "feedback"
                if t==1 and (feedback / "start.json").exists() and not (feedback / "complete.json").exists():
                    pending.append({"run":run,"T":2,"source_round":1,"candidate":name,"slot":candidate.get("slot"),"phase":"feedback",
                                    "completed":len(list((feedback / "items").glob("*.json"))),"count_precision":"saved_items","started":True,
                                    "error":read_json(feedback / "failure.json") if (feedback / "failure.json").exists() else None})
                for phase in ["train","score","audit"]:
                    folder=ROOT / "external/audit" / run / name if phase=="audit" else ROOT / "runs" / run / "text_classification/history" / name / phase
                    if (folder / "complete.json").exists():
                        continue
                    count=logged_training_count(folder) if phase=="train" else len(list((folder / "items").glob("*.json")))
                    pending.append({"run":run,"T":t,"candidate":name,"slot":candidate.get("slot"),"phase":phase,
                                    "completed":count,"count_precision":"last_logged_multiple_of_10" if phase=="train" else "saved_items",
                                    "started":(folder / "start.json").exists(),
                                    "error":read_json(folder / "failure.json") if (folder / "failure.json").exists() else None})
                    break
    now=datetime.now(timezone.utc)
    injected_path=ROOT / "external/injected_budget.json"
    injected=read_json(injected_path) if injected_path.exists() else None
    proposer_budget=None if injected is None else {
        "spent_usd":round(sum(r.get("cost_usd",0) for r in injected["requests"].values()),6),
        "reserved_usd":round(sum(r["reserved_usd"] for r in injected["requests"].values() if "cost_usd" not in r),6),
        "notice_threshold_usd":injected["cap_usd"],"thresholds_reached":injected["notified_thresholds"]}
    return {"utc":now.strftime("%H:%M:%S"),"local_time":now.astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S"),"status":read_json(ROOT / "external/status.json"),
            "known_usd":round(budget["known_or_estimated_usd"],4),"reserved_usd":round(budget["reserved_or_unknown_usd"],4),
            "injected_proposer_budget":proposer_budget,
            "checkpoints":len(list((ROOT / "external/control").glob("*/round*_checkpoint.json"))),"memory":memory_status(),"unfinished":pending}


if __name__=="__main__":
    print(json.dumps(progress()))
