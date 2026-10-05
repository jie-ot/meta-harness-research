"""Refresh the implementation patch and locally verify delivered artifacts."""
from pathlib import Path
import difflib
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pilot.state import atomic_json, digest, read_json, read_jsonl, utc_now


def main():
    verified = read_json(ROOT / "analysis/output_verification.json")
    assert verified["final"] and verified["status"] == "PASS"
    source = ROOT.parent / "reference_examples/text_classification"
    inventory = read_json(ROOT / "preflight.json")["source_inventory_sha256"]
    assert all(digest(source / rel) == sha for rel, sha in inventory.items())
    implementation = list((ROOT / "engine").rglob("*.py")) + list((ROOT / "pilot").glob("*.py"))
    implementation += [ROOT / "engine/text_classification/.claude/skills/meta-harness/SKILL.md",
                       ROOT / "engine/text_classification/config.yaml", ROOT / "preflight.py",
                       ROOT / "run.ps1", ROOT / "analyze.ps1", ROOT / "README.md", ROOT / "pilot_config.json", ROOT / "实施说明.md"]
    implementation += sorted((ROOT / "external/tools").glob("*.py"))
    patch, changed = [], []
    for path in sorted(set(implementation)):
        rel = path.relative_to(ROOT).as_posix()
        baseline = source / path.relative_to(ROOT / "engine/text_classification") if path.is_relative_to(ROOT / "engine/text_classification") else None
        before = baseline.read_text(encoding="utf-8").splitlines(keepends=True) if baseline and baseline.is_file() else []
        after = path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = list(difflib.unified_diff(before, after, fromfile="a/" + rel if before else "/dev/null", tofile="b/" + rel))
        if diff:
            changed.append(rel)
            patch.extend(diff)
    (ROOT / "code_changes.patch").write_text("".join(patch), encoding="utf-8")
    # Values stay in this process. Only the aggregate match count is saved.
    from dotenv import dotenv_values
    credentials = dotenv_values(ROOT.parent / ".env")
    needles = [credentials[k].encode("utf-8") for k in ["ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"] if credentials.get(k)]
    assert len(needles) == 2 and all(len(n) > 8 for n in needles)
    overlap = max(map(len, needles)) - 1
    def has_secret(handle):
        tail=b""
        for block in iter(lambda:handle.read(1024*1024),b""):
            data=tail+block
            if any(n in data for n in needles):return True
            tail=data[-overlap:]
        return False
    checked, matches = 0, []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc", ".lock", ".png"}:
            continue
        checked += 1
        with path.open("rb") as handle:
            if has_secret(handle):matches.append(path.relative_to(ROOT).as_posix())
        if path.suffix==".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():continue
                    checked+=1
                    with archive.open(member) as handle:
                        if has_secret(handle):matches.append(path.relative_to(ROOT).as_posix()+"!/"+member.filename)
    if matches:
        # Do not reveal matching text or place a contaminated artifact in an archive.
        atomic_json(ROOT / "external/secret_scan_review.json", {"status": "REVIEW_REQUIRED", "files_checked": checked, "matching_files": matches})
        raise RuntimeError("Credential value matched a local artifact; review required before delivery")
    old_audit = read_json(ROOT / "implementation_audit.json")
    final = read_json(ROOT / "analysis/analysis.json")
    accounting = read_json(ROOT / "analysis/accounting_review.json")
    archived = read_json(ROOT / "external/archive_manifest.json") if (ROOT / "external/archive_manifest.json").exists() else None
    ledger = read_jsonl(ROOT / "external/cost_ledger.jsonl")
    receipt = {
        "record_type": "completed_experiments_1_2_3", "updated_at_utc": utc_now(),
        "git_commit": old_audit["git_commit"], "source_worktree_was_already_dirty": True,
        "original_selected_files_unchanged": True, "original_files_verified": len(inventory),
        "changed_or_added_implementation_files": changed, "patch_sha256": digest(ROOT / "code_changes.patch"),
        "implementation_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(implementation))},
        "offline_tests_passed": read_json(ROOT / "verification.json")["tests_run"],
        "completed_experiments": final["completed_experiments"], "deferred_experiments": [4],
        "nominal_new_candidates": final["nominal_new_candidates"], "valid_new_candidates": final["valid_new_candidates"],
        "formal_proposer_sessions": final["proposer_sessions"],
        "actual_paid_proposer_attempts": accounting["actual_paid_proposer_attempts"],
        "solver_api_attempts_including_failures": sum(r["event"] == "api_started" for r in ledger),
        "solver_api_results": sum(r["event"] == "api_result" for r in ledger),
        "budget_usd": 80, "budget_snapshot": accounting["budget"],
        "secret_scan": {"files_checked": checked, "credential_value_matches": 0, "scope": "local exact-value scan including decompressed archive members; no upload"},
        "output_verification_sha256": digest(ROOT / "analysis/output_verification.json"),
        "user_format_hint_exception": ("external/运行过程归档.zip!/" if archived else "")+"external/user_format_hint_waiver.json" if archived else "external/user_format_hint_waiver.json",
        "history_archive": {"path":archived["archive_relative_path"],"sha256":archived["archive_sha256"],"files":len(archived["files"])} if archived else None,
        "source_and_failure_history_retained": True,
    }
    atomic_json(ROOT / "implementation_audit.json", receipt)
    print({"completed_experiments": [1, 2, 3], "original_files_unchanged": len(inventory),
           "secret_scan_files": checked, "credential_value_matches": 0, "patch_sha256": receipt["patch_sha256"]})


if __name__ == "__main__":
    main()
