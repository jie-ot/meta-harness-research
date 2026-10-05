"""Count actual proposer HTTP attempts while forwarding unchanged to its provider.

Only listens on loopback. Does not persist request headers, credentials or bodies.
The Claude CLI still determines model parameters, tools, retries and stream parsing.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from pilot.state import utc_now

HOP_HEADERS = {"connection", "transfer-encoding", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade", "host"}


class ProposerRelay:
    def __init__(self, upstream: str, events_path: Path, guard=None):
        self.upstream = urlsplit(upstream.rstrip("/"))
        if self.upstream.scheme != "https" and self.upstream.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Proposer upstream must use HTTPS; HTTP loopback is reserved for offline tests")
        if self.upstream.username or self.upstream.password or self.upstream.query:
            raise ValueError("Proposer upstream credentials must be headers, not URL components")
        self.events_path = events_path
        self.guard = guard
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connections = set()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *args): pass
            def do_GET(self): self.forward()
            def do_POST(self): self.forward()
            def forward(self):
                request_id = uuid.uuid4().hex
                start = time.monotonic()
                incoming = urlsplit(self.path)
                if incoming.scheme or incoming.netloc or not incoming.path.startswith("/v1/"):
                    self.send_error(400, "Only relative provider API paths are allowed")
                    return
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(400, "A known JSON content length is required")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else None
                model = None
                payload = None
                if body:
                    try:
                        payload = json.loads(body)
                        model = payload.get("model")
                    except (ValueError, AttributeError): pass
                guarded = bool(relay.guard and incoming.path == "/v1/messages" and self.command == "POST")
                if guarded:
                    try:
                        relay.guard.before(request_id, payload)
                        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        length = len(body)
                    except Exception as exc:
                        relay.record({"event": "request_blocked_before_upstream", "request_id": request_id,
                                      "reason": str(exc), "request_body_sha256": hashlib.sha256(body or b"").hexdigest()})
                        self.send_error(400, "Supplementary experiment pre-request check failed")
                        return
                path = relay.upstream.path + self.path
                headers = {k: v for k,v in self.headers.items() if k.lower() not in HOP_HEADERS}
                if guarded:
                    headers = {k:v for k,v in headers.items() if k.lower() != "content-length"}
                    headers["Content-Length"] = str(length)
                connection_class = http.client.HTTPSConnection if relay.upstream.scheme == "https" else http.client.HTTPConnection
                connection = connection_class(relay.upstream.hostname, relay.upstream.port, timeout=600)
                event = {"request_id": request_id, "method": self.command, "path": incoming.path,
                         "upstream_host": relay.upstream.hostname, "requested_model": model,
                         "request_body_sha256": hashlib.sha256(body or b"").hexdigest(), "request_bytes": length}
                with relay.lock: relay.connections.add(connection)
                relay.record({**event, "event": "http_started"})
                sent_headers = False
                response_bytes = 0
                status = None
                usage, sse_buffer, response_complete = {}, b"", False
                try:
                    connection.request(self.command, path, body=body, headers=headers)
                    response = connection.getresponse()
                    status = response.status
                    self.send_response(status)
                    for key, value in response.getheaders():
                        if key.lower() not in HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sent_headers = True
                    while True:
                        chunk = response.read1(65536)
                        if not chunk: break
                        if guarded:
                            sse_buffer += chunk
                            while b"\n" in sse_buffer:
                                line, sse_buffer = sse_buffer.split(b"\n", 1)
                                if not line.startswith(b"data: "):
                                    continue
                                try:
                                    entry = json.loads(line[6:])
                                    if entry.get("type") == "message_start":
                                        usage.update(entry.get("message", {}).get("usage", {}))
                                    elif entry.get("type") == "message_delta":
                                        usage.update(entry.get("usage", {}))
                                    elif entry.get("type") == "message_stop":
                                        response_complete = True
                                except (ValueError, AttributeError):
                                    pass
                        response_bytes += len(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    relay.record({**event, "event": "http_finished", "http_status": status,
                                  "response_bytes": response_bytes, "wall_seconds": time.monotonic() - start})
                    if guarded:
                        relay.guard.after(request_id, usage, response_complete, status)
                except Exception as exc:
                    relay.record({**event, "event": "http_failed", "http_status": status,
                                  "error_type": type(exc).__name__, "response_bytes": response_bytes,
                                  "wall_seconds": time.monotonic() - start})
                    if not sent_headers:
                        try: self.send_error(502, "Provider transport failure")
                        except OSError: pass
                finally:
                    self.close_connection = True
                    connection.close()
                    with relay.lock: relay.connections.discard(connection)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)

    def record(self, event):
        with self.lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({**event, "timestamp_utc": utc_now()}, ensure_ascii=False) + "\n")
                handle.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        with self.lock:
            for connection in self.connections:
                connection.close()


def summarize_requests(events):
    starts = [e for e in events if e["event"] == "http_started"]
    messages = [e for e in starts if e["method"] == "POST" and e["path"] == "/v1/messages"]
    ends = {e["request_id"]: e for e in events if e["event"] in {"http_finished", "http_failed"}}
    failed = sum(e["request_id"] not in ends or ends[e["request_id"]]["event"] == "http_failed" or (ends[e["request_id"]].get("http_status") or 0) >= 400 for e in messages)
    hashes = [e["request_body_sha256"] for e in messages]
    return {"api_requests": len(messages), "other_api_requests": len(starts) - len(messages),
            "failed_api_requests": failed, "repeated_request_body_attempts": len(hashes) - len(set(hashes)),
            "requested_models": sorted({e["requested_model"] for e in messages if e["requested_model"]}),
            "request_count_source": "loopback relay forwarded HTTP attempts to the original configured upstream"}
