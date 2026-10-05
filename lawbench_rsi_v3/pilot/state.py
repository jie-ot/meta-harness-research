from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_THREAD_LOCK = threading.RLock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part-" + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Windows readers or file scanners can briefly prevent replacing an open
    # checkpoint. Keep the old committed file intact and retry only the rename;
    # a persistent denial still fails with the fully written temp file retained.
    for attempt in range(7):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.025 * 2 ** attempt)


def write_once(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def jsonl_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part-" + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def exclusive_controller(path: Path):
    """Reject a second controller; do not hold the in-process ledger mutex."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another pilot controller is active") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another pilot controller is active") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def serial_stage(path: Path):
    """Wait for this exact stage; do not hold the ledger's thread mutex."""
    while True:
        stack = ExitStack()
        try:
            stack.enter_context(exclusive_controller(path))
        except RuntimeError:
            stack.close()
            time.sleep(0.25)
        else:
            break
    with stack:
        yield


class Ledger:
    """One controller ledger for paid calls, including failures and reservations."""
    def __init__(self, path: Path, budget: float):
        self.path = path
        self.budget = budget
        self.lock_path = path.with_suffix(".lock")

    def _summary(self, events):
        cost = 0.0
        reservations = {}
        for event in events:
            key = event["attempt_id"]
            if event["event"] in {"api_started", "proposer_started"}:
                reservations[key] = event["reserved_usd"]
            elif event["event"] in {"api_result", "proposer_result"}:
                reservations.pop(key, None)
                cost += float(event.get("cost") or 0)
            elif event["event"] in {"api_failed", "proposer_failed"}:
                if event.get("http_status") == 402:
                    # OpenRouter rejects credit checks before model generation.
                    # Keep the failure event, but release the unspent reservation.
                    reservations.pop(key, None)
                # Unreported failed-request charges keep their conservative reservation.
                pass
        return {"known_or_estimated_usd": cost, "reserved_or_unknown_usd": sum(reservations.values()), "outstanding_ids": list(reservations)}

    def summary(self):
        with file_lock(self.lock_path):
            return self._summary(read_jsonl(self.path))

    def add(self, event: dict):
        event = {**event, "timestamp_utc": utc_now()}
        with file_lock(self.lock_path):
            rows = read_jsonl(self.path)
            if any(r["event_id"] == event["event_id"] for r in rows):
                return
            if event["event"].endswith("_started"):
                total = self._summary(rows)
                committed = total["known_or_estimated_usd"] + total["reserved_or_unknown_usd"]
                # The authorized supplement has a separate proposer notice
                # threshold. Preserve the original runs' historical USD80 cap.
                injected = str(event.get("run_id", "")).startswith("D100_injected_")
                if not injected and committed + event["reserved_usd"] > self.budget:
                    raise BudgetExceeded("Budget exhausted; checkpoints retained")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def observe_api(self, event):
        if event["event"] not in {"api_started", "api_result", "api_failed"}:
            return
        allowed = ("event", "attempt_id", "call_id", "phase", "run_id", "candidate", "round", "D", "item_id", "cost", "reported_usd", "estimated_usd", "cost_source", "input_tokens", "output_tokens", "provider_usage", "wall_seconds", "error_type", "http_status")
        entry = {k: event[k] for k in allowed if k in event}
        entry["event_id"] = f"{event['attempt_id']}:{event['event']}"
        if event["event"] == "api_started":
            # Conservative reservation above the listed solver cost at the allowed context limit.
            entry["reserved_usd"] = 0.25
        self.add(entry)


def usage_from_calls(path: Path):
    rows = read_jsonl(path)
    results = [r for r in rows if r["event"] == "api_result"]
    failures = [r for r in rows if r["event"] == "api_failed"]
    started = {r["attempt_id"] for r in rows if r["event"] == "api_started"}
    ended = {r["attempt_id"] for r in rows if r["event"] in {"api_result", "api_failed"}}
    return {
        "logical_calls": len({r["call_id"] for r in rows}),
        "api_requests": len(started),
        "cache_hits": sum(r["event"] == "cache_hit" for r in rows),
        "input_tokens": sum(r.get("input_tokens", 0) for r in results),
        "output_tokens": sum(r.get("output_tokens", 0) for r in results),
        "cost_usd": sum(r.get("cost", 0) for r in results),
        "reported_usd": sum(r.get("reported_usd") or 0 for r in results),
        "estimated_usd": sum(r.get("estimated_usd") or 0 for r in results),
        "failed_api_requests": len(failures),
        "unreported_failed_cost_count": sum(r.get("http_status") != 402 for r in failures),
        "nonbillable_credit_rejections": sum(r.get("http_status") == 402 for r in failures),
        "incomplete_attempts": sorted(started - ended),
        "retry_attempts": sum(int(r.get("attempt", 0)) > 0 for r in rows if r["event"] == "api_started"),
    }


def verify_done(directory: Path, signature: dict):
    marker = directory / "complete.json"
    if not marker.exists():
        return None
    done = read_json(marker)
    if done["signature"] != signature:
        raise ValueError(f"Completed stage signature changed: {directory}")
    for rel, expected in done["files"].items():
        if digest(directory / rel) != expected:
            raise ValueError(f"Completed artifact changed: {directory / rel}")
    return done


def finish(directory: Path, signature: dict, files: list[str], **metadata):
    done = {"signature": signature, "files": {p: digest(directory / p) for p in files}, "completed_at_utc": utc_now(), **metadata}
    atomic_json(directory / "complete.json", done)
    return done
