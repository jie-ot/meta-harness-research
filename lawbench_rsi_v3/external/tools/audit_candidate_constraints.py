"""Read-only scan supporting manual review of the original generality rule."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pilot.state import atomic_json, digest, read_jsonl, utc_now


def audit():
    labels = {label.strip() for split in ["train", "score"]
              for row in read_jsonl(ROOT / f"external/prepared_data_v3/{split}.jsonl")
              for label in row["answer"].split(";") if len(label.strip()) >= 2}
    findings = []
    for run in ["D0_a", "D0_b", "D100_a", "D100_b"]:
        for path in (ROOT / "runs" / run / "text_classification/agents").glob("*.py"):
            if path.stem in {"confusion_disambiguation_memory", "__init__"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                        docs.add(id(first.value))
            hits = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                    found = sorted(label for label in labels if label in node.value)
                    if found:
                        hits.append({"line": node.lineno, "matched_labels": found})
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            imports += [item.name for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names]
            findings.append({"run": run, "candidate": path.stem, "code_hash": digest(path),
                             "executable_label_literal_hits": hits, "imports": imports,
                             "invalid_marked": (ROOT / "external/control" / run / "invalid" / f"{path.stem}.json").exists()})
    report = {"created_at_utc": utc_now(), "label_source": "shared train and score only; no audit labels", "candidates": findings,
              "interpretation": "Literal matches flag code for review; this scan does not measure quality or generate replacements."}
    atomic_json(ROOT / "external/candidate_constraint_audit.json", report)
    return report


if __name__ == "__main__":
    result = audit()
    print({"candidates":len(result["candidates"]), "candidates_with_literal_matches":sum(bool(r["executable_label_literal_hits"]) for r in result["candidates"])})
