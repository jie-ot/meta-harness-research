"""Thread-safe, per-item observations. No additional model requests."""

from __future__ import annotations

import contextvars
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_ACTIVE = contextvars.ContextVar("pilot_item", default=None)
_LOCK = threading.Lock()
_EVENT_CALLBACK = None


def set_event_callback(callback):
    global _EVENT_CALLBACK
    _EVENT_CALLBACK = callback


@contextmanager
def capture_calls(metadata: dict, sink: Path | None = None):
    captured = []
    token = _ACTIVE.set((dict(metadata), captured, sink))
    try:
        yield captured
    finally:
        _ACTIVE.reset(token)


def record_call(event: dict) -> None:
    active = _ACTIVE.get()
    if active is None:
        return
    metadata, captured, sink = active
    entry = {**metadata, **event, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if _EVENT_CALLBACK:
        _EVENT_CALLBACK(entry)
    with _LOCK:
        captured.append(entry)
        if sink:
            with Path(sink).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def submit_with_context(executor, function, *args):
    return executor.submit(contextvars.copy_context().run, function, *args)
