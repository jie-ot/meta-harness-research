"""Local, read-only checks against the prior experiment; no model calls."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "reference_examples" / "text_classification"
OLD_RUN = SOURCE / "logs" / "20260925_200856"
FROZEN = {
    "R4B": "confusion_disambiguation_memory",
    "R7A": "adaptive_tokenizer_confusion_memory",
    "R12B": "discriminative_similarity_memory",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    import yaml
    from dotenv import dotenv_values

    cfg = yaml.safe_load((SOURCE / "config.yaml").read_text(encoding="utf-8"))
    inventory = {}
    for path in sorted(SOURCE.glob("*.py")):
        inventory[str(path.relative_to(SOURCE))] = sha256(path)
    for path in sorted((SOURCE / "data").glob("*.py")):
        inventory[str(path.relative_to(SOURCE))] = sha256(path)
    for rel in ["config.yaml", "pyproject.toml", "uv.lock", ".claude/skills/meta-harness/SKILL.md"]:
        inventory[rel] = sha256(SOURCE / rel)

    splits = {}
    rows_by_split = {}
    cases = {}
    for split, expected in [("train", 200), ("val", 50), ("test", 100)]:
        path = SOURCE / "data" / "crime_prediction" / f"{split}.jsonl"
        rows = read_jsonl(path)
        rows_by_split[split] = rows
        ids = [hashlib.sha256(row["question"].strip().encode("utf-8")).hexdigest() for row in rows]
        cases[split] = set(ids)
        assert len(rows) == len(set(ids)) == expected, (split, len(rows), len(set(ids)))
        inventory[str(path.relative_to(SOURCE))] = sha256(path)
        splits[split] = {"rows": len(rows), "unique_cases": len(set(ids)), "sha256": sha256(path)}
    overlaps = {f"{a}/{b}": len(cases[a] & cases[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    assert not any(overlaps.values()), overlaps

    # Match the actual old execution order against all 200 stored training steps.
    ordered = list(rows_by_split["train"])
    random.Random(cfg["inner_loop"]["seed"]).shuffle(ordered)
    train_order_ids = [hashlib.sha256(row["question"].encode("utf-8")).hexdigest() for row in ordered]
    frozen = {}
    for label, name in FROZEN.items():
        code = SOURCE / "agents" / f"{name}.py"
        base = OLD_RUN / "LawBench" / name / "gpt-oss-120b"
        memory = base / "memory.json"
        log = base / "log.jsonl"
        steps = [row for row in read_jsonl(log) if row.get("type") == "step"]
        assert len(steps) == 200 and [row["step"] for row in steps] == list(range(200)), name
        state = json.loads(memory.read_text(encoding="utf-8"))
        examples = state["examples"]
        assert len(examples) == 200, (name, len(examples))
        matches = [ex.get("raw_question") == row["question"] for ex, row in zip(examples, ordered)]
        if not all(matches):
            # Some implementations retain the full wrapped input instead of raw_question.
            matches = [ex.get("input", "").endswith(row["question"]) for ex, row in zip(examples, ordered)]
        assert all(matches), (name, "saved memory order differs from original seed order")
        for path in [code, memory, log]:
            inventory[str(path.relative_to(SOURCE))] = sha256(path)
        frozen[label] = {"name": name, "code_sha256": sha256(code), "memory_sha256": sha256(memory), "log_sha256": sha256(log), "train_steps": len(steps), "memory_train_order_matches": True}

    # Inspect variable names and presence only. Never serialize .env values or hashes.
    pilot_config = json.loads((ROOT / "pilot_config.json").read_text(encoding="utf-8"))
    env = dotenv_values((ROOT / pilot_config["credentials_file"]).resolve())
    credential_presence = {key: bool(env.get(key)) for key in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"]}
    versions = {}
    for package in ["litellm", "openai-harmony", "datasets", "tenacity", "python-dotenv", "PyYAML"]:
        versions[package] = importlib.metadata.version(package)
    tree = ast.parse((SOURCE / "meta_harness.py").read_text(encoding="utf-8"))
    proposer = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "propose_claude":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "run":
                    for kw in call.keywords:
                        if kw.arg in {"model", "effort"}:
                            proposer[kw.arg] = ast.literal_eval(kw.value)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "local_source_checks_passed",
        "model_api_requests": 0,
        "source_dir": str(SOURCE),
        "source_run": str(OLD_RUN),
        "source_inventory_sha256": inventory,
        "original_splits": splits,
        "original_case_overlap": overlaps,
        "training_order": {"seed": cfg["inner_loop"]["seed"], "case_ids_in_execution_order": train_order_ids},
        "frozen": frozen,
        "solver_models": cfg["models"],
        "inner_loop": cfg["inner_loop"],
        "proposer": proposer,
        "credentials_present": credential_presence,
        "python": sys.version,
        "dependency_versions": versions,
        "data_audit": json.loads((ROOT / "external/prepared_data_v3/data_audit.json").read_text(encoding="utf-8")) if (ROOT / "external/prepared_data_v3/data_audit.json").exists() else None,
        "pending": ["Live generation and provider cost verification", "User-authorized paid experiments and analysis"],
    }
    output = ROOT / "preflight.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        assert previous["source_inventory_sha256"] == inventory, "Source changed since preflight; stop for review."
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "model_api_requests": 0, "splits": splits, "frozen": frozen, "solver": cfg["models"], "proposer": proposer, "credentials_present": credential_presence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
