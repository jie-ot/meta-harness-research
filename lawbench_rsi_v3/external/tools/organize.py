"""Consolidate completed pilot history, preserving every byte in a verified ZIP.

Without --apply this only inventories the proposed archive. Formal run data,
candidate code, the ledger, caches, dataset and test paths remain in place.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pilot.state import atomic_json, exclusive_controller, read_json, utc_now

DIRECTORIES=["cli_acceptance","diagnostic_acceptance_source","failures","implementation_revisions","validation_recovery"]
FILES=["continue_unaffected_round1.py","recover_saved_d0_b_round2.py","restore_descriptor_candidates.py","restore_user_waived_candidate.py",
       "continuation_state.json","d0b_recovery_launch.json","historical_start_cost.json","launch.json",
       "openrouter_billing_diagnostic.json","openrouter_billing_preflight.json","protocol_observations.json",
       "prototype_filename_review.json","response_recording_scope.json","unaffected_round1_launch.json",
       "unaffected_round1_status.json","unaffected_round1.lock","user_format_hint_waiver.json",
       "validation_false_positive_review.json","validation_repair_tests.txt","validation_repair_verification.json"]
ARCHIVE=ROOT / "external/运行过程归档.zip"
MANIFEST=ROOT / "external/archive_manifest.json"


def checksum_stream(handle):
    value=hashlib.sha256()
    for block in iter(lambda:handle.read(1024*1024),b""):
        value.update(block)
    return value.hexdigest()


def checksum(path):
    with path.open("rb") as handle:
        return checksum_stream(handle)


def contained(path):
    resolved=path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Path escapes this experiment directory")
    return resolved


def inventory():
    external=ROOT / "external"
    paths=[]
    for name in DIRECTORIES:
        directory=external / name
        if directory.exists():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    paths.extend(external / name for name in FILES if (external / name).is_file())
    for pattern in ["*.out.log","*.err.log","launch_*.json"]:
        paths.extend(external.glob(pattern))
    rows=[]
    for path in sorted(set(paths)):
        if path.is_symlink():
            raise ValueError("Archive review required for symlink")
        contained(path)
        rows.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":checksum(path)})
    return rows


def verify_archive(record):
    if checksum(ARCHIVE)!=record["archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(ARCHIVE) as archive:
        expected={row["path"] for row in record["files"]}|{"MANIFEST.json"}
        if set(archive.namelist())!=expected:
            raise ValueError("Archive members differ from the manifest")
        for row in record["files"]:
            with archive.open(row["path"]) as handle:
                if checksum_stream(handle)!=row["sha256"]:
                    raise ValueError("Archived file checksum mismatch")


def remove_verified_source_copies(record):
    # Only files named in the byte-verified archive can be retired. Check again
    # immediately before unlinking, and leave any unexpected new files intact.
    for row in record["files"]:
        path=contained(ROOT / row["path"])
        if path.exists() and (not path.is_file() or checksum(path)!=row["sha256"]):
            raise ValueError("Source changed after archiving; preserve it for review")
    for row in record["files"]:
        path=contained(ROOT / row["path"])
        if path.exists():
            if checksum(path)!=row["sha256"]:
                raise ValueError("Source changed before retirement")
            path.unlink()
    for name in DIRECTORIES:
        directory=contained(ROOT / "external" / name)
        if directory.exists():
            for path in sorted((p for p in directory.rglob("*") if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                if not any(path.iterdir()):contained(path).rmdir()
            if not any(directory.iterdir()):directory.rmdir()
    temporary=contained(ROOT / ".verification_tmp")
    if temporary.exists() and not any(temporary.iterdir()):temporary.rmdir()


def organize(apply=False):
    if not apply:
        rows=inventory()
        print(json.dumps({"mode":"inventory_only","files":len(rows),"uncompressed_MiB":round(sum(r['bytes'] for r in rows)/1024**2,2),"archive":str(ARCHIVE)},ensure_ascii=False))
        return
    verified=read_json(ROOT / "analysis/output_verification.json")
    assert verified["final"] and verified["status"]=="PASS"
    assert read_json(ROOT / "analysis/analysis.json")["completed_experiments"]==[1,2,3]
    audit=read_json(ROOT / "implementation_audit.json")
    assert audit["record_type"]=="completed_experiments_1_2_3" and audit["secret_scan"]["credential_value_matches"]==0
    if MANIFEST.exists():
        record=read_json(MANIFEST)
    else:
        assert not ARCHIVE.exists(), "Existing unregistered archive needs review"
        rows=inventory()
        record={"created_at_utc":utc_now(),"files":rows,"scope":"Closed execution/development/failure records only; measurement and resume paths unchanged.","secret_scan_before_archiving":audit["secret_scan"]}
        temporary=ARCHIVE.with_suffix(".zip.part")
        assert not temporary.exists(), "Existing partial archive needs review"
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as archive:
            for row in rows:
                source=contained(ROOT / row["path"])
                if checksum(source)!=row["sha256"]:raise ValueError("Source changed during archive preparation")
                archive.write(source,arcname=row["path"])
            archive.writestr("MANIFEST.json",json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        temporary.replace(ARCHIVE)
        record.update({"archive_sha256":checksum(ARCHIVE),"archive_bytes":ARCHIVE.stat().st_size,"archive_relative_path":ARCHIVE.relative_to(ROOT).as_posix()})
        atomic_json(MANIFEST,record)
    verify_archive(record)
    remove_verified_source_copies(record)
    record["source_copies_retired_at_utc"]=utc_now()
    atomic_json(MANIFEST,record)
    print(json.dumps({"archived_files":len(record["files"]),"archive_MiB":round(record['archive_bytes']/1024**2,2),"every_member_sha256_verified":True,"measurement_paths_unchanged":True},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--apply",action="store_true");args=parser.parse_args()
    if args.apply:
        with exclusive_controller(ROOT / "external/controller.lock"),exclusive_controller(ROOT / "external/trajectory_recovery.lock"):
            organize(True)
    else:
        organize(False)
