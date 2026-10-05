"""Resume the frozen E4 repeat under the authorized infrastructure-retry policy.

This program does not import project modules.  It sends only the already locked
prompt text through the official Codex CLI app-server, one ephemeral thread at a
time.  Every model attempt is preceded by a read of the seven-day Codex bucket.
"""

import datetime
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


OUT = Path(__file__).resolve().parent
RESUME = OUT / "resume_observations"
NODE = r"D:\DevEnvs\Nodejs\node.exe"
CLI = r"D:\DevEnvs\Nodejs\node_modules\@openai\codex\bin\codex.js"
MODEL = "gpt-6.1-sol"
EFFORT = "max"
RETRY_PROTOCOL_HASH = "b056c3f7671961916865e0f3fe6318db97d080ec8342a5a882793c2d10312a9e"
BATCH_POLICY_HASH = "6fc42365443a7b759dcec920ceaf4e355638aea1382f6698c567f6c049cd6e2a"
REQUEST_CONFIG_HASH = "25cb509c3b49fb2cd4967df77aeb148693fd0cddf039611382c00f2fa740cd4b"
DEADLINE = datetime.datetime(2026, 10, 5, 14, 11, tzinfo=datetime.timezone.utc)
STOP_REMAINING = 2
PROCESS_TIMEOUT_SECONDS = 3600
TURN_TIMEOUT_SECONDS = 300
DISABLED_FEATURES = [
    "shell_tool", "unified_exec", "shell_snapshot", "memories", "apps",
    "browser_use", "computer_use", "image_generation", "multi_agent",
    "plugins", "view_image", "browser_use_external", "browser_use_full_cdp_access",
    "in_app_browser", "multi_agent_v2", "sleep_tool", "hooks",
]
TOOL_TYPES = {
    "commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall",
    "collabAgentToolCall", "webSearch", "imageView", "imageGeneration",
}
SENSITIVE_KEY_PARTS = ("token", "authorization", "credential", "cookie", "api_key", "email")
REDACTED = "[REDACTED]"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_digest(text):
    return digest(text.encode("utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sanitize(value, parent_key=""):
    """Recursively retain error structure while suppressing credential-like values."""
    key_lower = str(parent_key).lower()
    if any(fragment in key_lower for fragment in SENSITIVE_KEY_PARTS):
        return REDACTED
    if isinstance(value, dict):
        return {str(key): sanitize(child, str(key)) for key, child in value.items()}
    if isinstance(value, list):
        return [sanitize(child, parent_key) for child in value]
    if isinstance(value, tuple):
        return [sanitize(child, parent_key) for child in value]
    if isinstance(value, str):
        cleaned = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/]+=*", "Bearer [REDACTED]", value)
        cleaned = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", cleaned)
        return cleaned
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def collect_codes(value):
    codes = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in {"code", "status", "statuscode", "status_code", "httpstatus", "http_status"}:
                if isinstance(child, int):
                    codes.append(child)
                elif isinstance(child, str) and child.isdigit():
                    codes.append(int(child))
            codes.extend(collect_codes(child))
    elif isinstance(value, list):
        for child in value:
            codes.extend(collect_codes(child))
    return codes


def classify_error_structures(structures):
    """Return only policy-relevant, non-retryable error classes."""
    sanitized = sanitize(structures)
    codes = set(collect_codes(sanitized))
    text = json.dumps(sanitized, ensure_ascii=False).lower()
    classes = []
    if codes.intersection({401, 403}) or re.search(
        r"\bunauthori[sz]ed\b|\bforbidden\b|\bauthentication\b|permission|not permitted|access denied",
        text,
    ):
        classes.append("permission_or_authentication")
    if codes.intersection({402}) or re.search(
        r"credit|quota|usage limit|depleted",
        text,
    ):
        classes.append("quota_or_credits_exhausted")
    if codes.intersection({429}) or re.search(r"rate[ _-]?limit|too many requests", text):
        classes.append("rate_limit")
    if re.search(
        r"model (?:is )?(?:unavailable|not available|not found|unsupported)|unknown model|does not support",
        text,
    ):
        classes.append("model_unavailable")
    if re.search(r"safety|moderation|content policy|policy refusal", text):
        classes.append("safety_or_moderation_refusal")
    return sorted(set(classes))


def retry_decision(turn_status, response_text, attempt_number, error_classes):
    if response_text:
        return "answer_present_no_retry"
    if error_classes:
        return "stop_non_retryable"
    if turn_status != "failed":
        return "stop_not_explicit_terminal_failed"
    if attempt_number < 2:
        return "retry_once"
    return "missing_after_retry_continue"


def extract_json_field(text, field, default=""):
    """Frozen copy of the evaluator's permissive JSON-field extraction."""
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return str(value.get(field, default))
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return str(value.get(field, default))
        except json.JSONDecodeError:
            pass
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth, pos, in_string = 1, start + 1, False
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == '"' and (pos == 0 or text[pos - 1] != "\\"):
                in_string = not in_string
            elif not in_string:
                depth += 1 if char == "{" else (-1 if char == "}" else 0)
            pos += 1
        if depth == 0:
            candidate = re.sub(r",\s*([\]}])", r"\1", text[start:pos])
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return str(value.get(field, default))
            except json.JSONDecodeError:
                pass
    matches = re.findall(rf'"{field}"\s*:\s*"([^"]*)"', text)
    return matches[-1] if matches else default


def canonical(text):
    text = text.strip()
    match = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text)
    if match:
        text = match.group(1).strip()
    elif "罪名:" in text:
        text = text.split("罪名:")[-1]
    text = re.sub(r"<eoa>.*", "", text).strip()
    for separator in [";", "；", ",", "，", "、"]:
        if separator in text:
            return sorted({part.strip() for part in text.split(separator) if part.strip()})
    return [text] if text else []


def weekly_snapshot(response, queried_at, received_at):
    by_id = response.get("rateLimitsByLimitId")
    value = by_id.get("codex") if isinstance(by_id, dict) and isinstance(by_id.get("codex"), dict) else response.get("rateLimits")
    if not isinstance(value, dict):
        return {"valid": False, "queried_at_utc": queried_at, "received_at_utc": received_at}
    primary = value.get("primary")
    if not isinstance(primary, dict):
        return {"valid": False, "queried_at_utc": queried_at, "received_at_utc": received_at}
    used = primary.get("usedPercent")
    duration = primary.get("windowDurationMins")
    return {
        "valid": type(used) is int and duration == 10080,
        "queried_at_utc": queried_at,
        "received_at_utc": received_at,
        "bucket_map_key": "codex" if isinstance(by_id, dict) and "codex" in by_id else None,
        "limit_id": value.get("limitId"),
        "window_duration_minutes": duration,
        "used_percent": used,
        "remaining_percent_derived": 100 - used if type(used) is int and 0 <= used <= 100 else None,
        "resets_at_epoch_seconds": primary.get("resetsAt"),
    }


class RunStop(Exception):
    def __init__(self, kind, method=None, detail=None):
        super().__init__(kind)
        self.kind = kind
        self.method = method
        self.detail = sanitize(detail)

    def record(self):
        return {"kind": self.kind, "method_or_observation": self.method, "detail": self.detail}


def selfcheck():
    sample = {
        "code": 429,
        "message": "rate limit with Bearer abc.def.ghi",
        "access_token": "secret",
        "nested": {
            "Authorization": "Bearer secret",
            "credentialHint": "secret",
            "cookieJar": "secret",
            "api_key": "secret",
            "emailAddress": "person@example.test",
            "keep": "visible",
        },
    }
    cleaned = sanitize(sample)
    checks = {
        "sensitive_keys_redacted": (
            cleaned["access_token"] == REDACTED
            and all(cleaned["nested"][key] == REDACTED for key in
                    ("Authorization", "credentialHint", "cookieJar", "api_key", "emailAddress"))
        ),
        "code_and_message_preserved": cleaned["code"] == 429 and cleaned["message"].startswith("rate limit"),
        "bearer_value_redacted": "abc.def.ghi" not in cleaned["message"],
        "429_non_retryable": classify_error_structures({"code": 429, "message": "busy"}) == ["rate_limit"],
        "auth_non_retryable": "permission_or_authentication" in classify_error_structures({"status": 403, "message": "forbidden"}),
        "quota_non_retryable": "quota_or_credits_exhausted" in classify_error_structures({"message": "quota exhausted"}),
        "model_non_retryable": "model_unavailable" in classify_error_structures({"message": "model is unavailable"}),
        "safety_non_retryable": "safety_or_moderation_refusal" in classify_error_structures({"message": "moderation refusal"}),
        "500_transient_unclassified": classify_error_structures({"code": 500, "message": "internal server error"}) == [],
        "failed_no_final_retryable": retry_decision("failed", "", 1, []) == "retry_once",
        "second_failed_no_final_exhausted": retry_decision("failed", "", 2, []) == "missing_after_retry_continue",
        "completed_text_never_retried": retry_decision("completed", "answer", 1, []) == "answer_present_no_retry",
        "failed_text_never_retried": retry_decision("failed", "answer", 1, []) == "answer_present_no_retry",
        "nonfailed_no_text_not_retryable": retry_decision("completed", "", 1, []) == "stop_not_explicit_terminal_failed",
    }
    report = {
        "checked_at_utc": utcnow(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "retry_protocol_sha256": digest((OUT / "retry_protocol.json").read_bytes()),
    }
    write_json(OUT / "resume_selfcheck.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


def main():
    marker = OUT / "resume_started.json"
    state_path = OUT / "resume_state.json"
    if marker.exists():
        raise SystemExit("Resume was already started. Automatic restart is forbidden.")

    protocol_bytes = (OUT / "retry_protocol.json").read_bytes()
    policy_bytes = (OUT / "batch_policy.json").read_bytes()
    config_bytes = (OUT / "request_config.json").read_bytes()
    if digest(protocol_bytes) != RETRY_PROTOCOL_HASH:
        raise SystemExit("Frozen retry protocol changed.")
    if digest(policy_bytes) != BATCH_POLICY_HASH or digest(config_bytes) != REQUEST_CONFIG_HASH:
        raise SystemExit("Frozen batch policy or request config changed.")

    config = json.loads(config_bytes)
    locked = [json.loads(line) for line in (OUT / "locked_items.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    wrapper = json.loads((OUT / "instruction_source_assessment.json").read_text(encoding="utf-8"))
    feature_check = json.loads((OUT / "tool_feature_precheck_0160.json").read_text(encoding="utf-8"))

    prerequisites = {
        "twenty_locked_items": len(locked) == 20,
        "requested_model_exact": config["thread_start"]["model"] == MODEL and config["turn_start"]["model"] == MODEL,
        "requested_effort_exact": config["turn_start"]["effort"] == EFFORT,
        "wrapper_previously_accepted": wrapper.get("accept_as_fixed_wrapper") is True,
        "wrapper_task_data_clean": wrapper.get("sources", [{}])[0].get("task_data_detected") is False,
        "temporary_tool_disables_prechecked": all(value is False for value in feature_check.get("effective_states", {}).values()),
    }
    if not all(prerequisites.values()):
        write_json(OUT / "resume_preflight_failure.json", {"checked_at_utc": utcnow(), "checks": prerequisites})
        raise SystemExit("Resume prerequisites failed; no model request was sent.")
    for item in locked:
        prompt_path = OUT / item["prompt_path"]
        if digest(prompt_path.read_bytes()) != item["prompt_sha256"]:
            raise SystemExit("A locked prompt byte hash changed; no model request was sent.")

    source_expected = wrapper["sources"][0]
    schedule = [(pass_number, index, 1) for pass_number in (1, 2) for index in range(20)]
    started_at = utcnow()
    state = {
        "status": "starting",
        "started_at_utc": started_at,
        "retry_protocol_sha256": RETRY_PROTOCOL_HASH,
        "model": MODEL,
        "effort": EFFORT,
        "quota_stop_remaining_lte": STOP_REMAINING,
        "schedule": [{"pass": p, "locked_item_index": i, "first_attempt_number": a} for p, i, a in schedule],
        "historical_attempts_before_resume": 0,
        "historical_completed_answers_before_resume": 0,
        "historical_missing_attempts_before_resume": 0,
        "new_turn_start_requests_sent": 0,
        "new_answer_observations": 0,
        "new_failed_no_final_attempts": 0,
        "new_infrastructure_retries_sent": 0,
        "items_finalized_in_resume": 0,
        "items_missing_after_retry": 0,
        "consecutive_final_missing_items": 0,
        "next_schedule_position": 0,
        "stop_reason": None,
        "attempts": [],
        "item_outcomes": [],
        "control_request_log": [],
        "model_substitutions": 0,
        "credit_resets_or_paid_api": 0,
    }
    write_json(marker, {
        "started_at_utc": started_at,
        "retry_protocol_sha256": RETRY_PROTOCOL_HASH,
        "schedule": state["schedule"],
        "prior_configuration_results": "../e4_live_repeat (gpt-5.6-sol/max), excluded from this batch",
    })
    write_json(state_path, state)
    RESUME.mkdir(exist_ok=True)

    proc = None
    cleanup = "not_started"
    messages = queue.Queue()
    request_id = 0
    process_deadline = time.monotonic() + PROCESS_TIMEOUT_SECONDS
    active = None

    def persist_state():
        write_json(state_path, state)

    def active_view():
        if active is None:
            return None
        return {key: value for key, value in active.items() if not key.startswith("_")}

    def persist_active(filename="progress.json"):
        if active is not None:
            write_json(active["_directory"] / filename, active_view())

    def request_log_target():
        return active["request_log"] if active is not None else state["control_request_log"]

    def send(method, params=None, include_params=True):
        nonlocal request_id
        request_id += 1
        request = {"id": request_id, "method": method}
        if include_params:
            request["params"] = params
        entry = {"id": request_id, "method": method, "sent_at_utc": utcnow(), "state": "sent"}
        request_log_target().append(entry)
        proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        return request_id, entry

    def send_notification(method, params=None, include_params=False):
        value = {"method": method}
        if include_params:
            value["params"] = params
        proc.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def deny_server_request(value):
        response = {
            "id": value.get("id"),
            "error": {"code": -32000, "message": "E4 adapter rejects all tool/client actions"},
        }
        proc.stdin.write(json.dumps(response, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    def handle(value):
        nonlocal active
        method = value.get("method")
        if not isinstance(method, str) or active is None:
            return
        params = value.get("params") if isinstance(value.get("params"), dict) else {}
        event = {"method": method, "received_at_utc": utcnow()}
        if method in ("item/started", "item/completed"):
            item_value = params.get("item") if isinstance(params.get("item"), dict) else {}
            item_type = item_value.get("type")
            event["item_type"] = item_type
            active["item_types"].append(item_type)
            if item_type in TOOL_TYPES:
                active["tool_violation"] = True
                event["protocol_violation"] = True
                event["item_sanitized"] = sanitize(item_value)
                if active.get("thread_id") and active.get("turn_id"):
                    send("turn/interrupt", {"threadId": active["thread_id"], "turnId": active["turn_id"]})
            if method == "item/completed" and item_type == "agentMessage":
                active["agent_messages"].append({
                    "id": item_value.get("id"),
                    "phase": item_value.get("phase"),
                    "text": item_value.get("text", ""),
                })
        elif method == "thread/tokenUsage/updated":
            active["token_usage"] = params.get("tokenUsage")
            event["usage_present"] = isinstance(active["token_usage"], dict)
        elif method == "thread/settings/updated":
            settings = params.get("threadSettings") if isinstance(params.get("threadSettings"), dict) else {}
            active["settings_echo"] = {
                key: settings.get(key) for key in
                ("model", "modelProvider", "effort", "summary", "approvalPolicy", "sandboxPolicy", "cwd")
            }
            event["settings_echo_recorded"] = True
        elif method == "turn/started":
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
            active["turn_id"] = turn.get("id") or active.get("turn_id")
            event["turn_id"] = active["turn_id"]
        elif method == "error":
            sanitized_error = sanitize(params)
            active["error_notifications"].append(sanitized_error)
            event["params_sanitized"] = sanitized_error
        elif method == "turn/completed":
            turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
            active["turn_complete"] = sanitize(turn)
            active["turn_id"] = turn.get("id") or active.get("turn_id")
            active["turn_completed_monotonic"] = time.monotonic()
            event["turn_status"] = turn.get("status")
            if "error" in turn:
                active["turn_error_sanitized"] = sanitize(turn.get("error"))
                event["turn_error_sanitized"] = active["turn_error_sanitized"]
        active["events"].append(event)
        persist_active("event_and_attempt_state.json")

    def rpc(method, params=None, include_params=True, timeout=20):
        identity, request_entry = send(method, params, include_params)
        end = min(process_deadline, time.monotonic() + timeout)
        while True:
            left = end - time.monotonic()
            if left <= 0:
                request_entry.update({"state": "timeout", "finished_at_utc": utcnow()})
                persist_active()
                persist_state()
                raise RunStop("rpc_timeout", method)
            try:
                value = messages.get(timeout=left)
            except queue.Empty:
                request_entry.update({"state": "timeout", "finished_at_utc": utcnow()})
                persist_active()
                persist_state()
                raise RunStop("rpc_timeout", method)
            if value is None:
                request_entry.update({"state": "stdio_process_exited", "finished_at_utc": utcnow()})
                persist_active()
                persist_state()
                raise RunStop("stdio_process_exited", method)
            if "method" in value and "id" in value:
                deny_server_request(value)
                if active is not None:
                    active["client_action_request_rejected"] = True
                    active["server_request_sanitized"] = sanitize(value)
                    persist_active()
                raise RunStop("server_requested_client_action", method, value)
            if "method" in value:
                handle(value)
                continue
            if value.get("id") != identity:
                continue
            request_entry["finished_at_utc"] = utcnow()
            if "error" in value:
                sanitized_error = sanitize(value.get("error"))
                request_entry.update({"state": "json_rpc_error", "error_sanitized": sanitized_error})
                if active is not None:
                    active["json_rpc_errors"].append({"method": method, "error": sanitized_error})
                    persist_active()
                else:
                    persist_state()
                raise RunStop("json_rpc_error", method, sanitized_error)
            if not isinstance(value.get("result"), dict):
                request_entry["state"] = "unexpected_result_shape"
                persist_active()
                persist_state()
                raise RunStop("unexpected_result_shape", method, sanitize(value))
            request_entry["state"] = "success"
            if active is not None:
                persist_active()
            else:
                persist_state()
            return value["result"]

    def drain_notifications(seconds=0.75):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                value = messages.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if value is None:
                raise RunStop("stdio_process_exited_during_notification_drain", "notification_drain")
            if "method" in value and "id" in value:
                deny_server_request(value)
                if active is not None:
                    active["client_action_request_rejected"] = True
                    active["server_request_sanitized"] = sanitize(value)
                raise RunStop("server_requested_client_action", "notification_drain", value)
            if "method" in value:
                handle(value)

    def run_attempt(pass_number, item_index, attempt_number, is_historical_retry):
        nonlocal active
        if datetime.datetime.now(datetime.timezone.utc) >= DEADLINE:
            raise RunStop("user_wallclock_deadline_reached", "before_attempt")
        item = locked[item_index]
        attempt_id = f"pass{pass_number}_item{item_index:02d}_attempt{attempt_number:02d}"
        directory = RESUME / attempt_id
        directory.mkdir(exist_ok=False)
        prompt = (OUT / item["prompt_path"]).read_bytes().decode("utf-8")
        active = {
            "attempt_id": attempt_id,
            "pass": pass_number,
            "locked_item_index": item_index,
            "item_id": item["item_id"],
            "attempt_number": attempt_number,
            "is_infrastructure_retry": attempt_number == 2,
            "retries_historical_failed_attempt": bool(is_historical_retry),
            "started_at_utc": utcnow(),
            "prompt_sha256": item["prompt_sha256"],
            "prompt_chars": item["prompt_chars"],
            "turn_start_sent": False,
            "thread_id": None,
            "turn_id": None,
            "turn_complete": None,
            "turn_error_sanitized": None,
            "thread_echo": None,
            "settings_echo": None,
            "events": [],
            "agent_messages": [],
            "item_types": [],
            "token_usage": None,
            "error_notifications": [],
            "json_rpc_errors": [],
            "request_log": [],
            "tool_violation": False,
            "client_action_request_rejected": False,
            "_directory": directory,
        }
        persist_active()
        if text_digest(prompt) != item["prompt_sha256"]:
            raise RunStop("prompt_hash_mismatch_before_request", attempt_id)

        query_time = utcnow()
        rate_before = weekly_snapshot(
            rpc("account/rateLimits/read", include_params=False, timeout=15), query_time, utcnow()
        )
        active["rate_before"] = rate_before
        persist_active()
        if not rate_before["valid"]:
            raise RunStop("weekly_rate_limit_snapshot_invalid", attempt_id)
        if rate_before["remaining_percent_derived"] <= STOP_REMAINING:
            reason = {
                "kind": "weekly_safety_margin_reached",
                "before_attempt": attempt_id,
                "remaining_percent": rate_before["remaining_percent_derived"],
            }
            write_json(directory / "not_sent.json", reason)
            active = None
            return {"outcome": "not_sent_quota", "stop_reason": reason}

        thread = rpc("thread/start", config["thread_start"], timeout=20)
        thread_value = thread.get("thread") if isinstance(thread.get("thread"), dict) else {}
        active["thread_id"] = thread_value.get("id")
        sources = thread.get("instructionSources") if isinstance(thread.get("instructionSources"), list) else []
        source_rows = []
        for source in sources:
            if not isinstance(source, str):
                source_rows.append({"valid": False})
                continue
            path = Path(source)
            normalized = str(path.resolve()) if path.is_absolute() else source
            source_rows.append({
                "valid": path.is_file(),
                "path_sha256": digest(normalized.encode("utf-8")),
                "content_sha256": digest(path.read_bytes()) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            })
        source_match = (
            len(source_rows) == 1
            and source_rows[0].get("valid") is True
            and source_rows[0].get("path_sha256") == source_expected["path_sha256"]
            and source_rows[0].get("content_sha256") == source_expected["content_sha256"]
            and source_rows[0].get("bytes") == source_expected["bytes"]
        )
        active["thread_echo"] = {
            "model": thread.get("model"),
            "model_provider": thread.get("modelProvider"),
            "reasoning_effort": thread.get("reasoningEffort"),
            "ephemeral": thread_value.get("ephemeral"),
            "approval_policy": thread.get("approvalPolicy"),
            "sandbox": thread.get("sandbox"),
            "instruction_source_count": len(sources),
            "wrapper_hash_match": source_match,
        }
        persist_active()
        if (
            not active["thread_id"]
            or thread.get("model") != MODEL
            or thread.get("reasoningEffort") != EFFORT
            or thread_value.get("ephemeral") is not True
            or not source_match
        ):
            raise RunStop("thread_protocol_mismatch", attempt_id, active["thread_echo"])

        turn_params = dict(config["turn_start"])
        turn_params.pop("input_rule", None)
        turn_params.pop("output_schema", None)
        turn_params["threadId"] = active["thread_id"]
        turn_params["input"] = [{"type": "text", "text": prompt}]
        if len(turn_params["input"]) != 1 or text_digest(turn_params["input"][0]["text"]) != item["prompt_sha256"]:
            raise RunStop("outgoing_prompt_mismatch", attempt_id)

        active["turn_start_sent"] = True
        active["turn_started_monotonic"] = time.monotonic()
        active["turn_started_at_utc"] = utcnow()
        state["new_turn_start_requests_sent"] += 1
        if attempt_number == 2:
            state["new_infrastructure_retries_sent"] += 1
        persist_active()
        persist_state()
        response = rpc("turn/start", turn_params, timeout=25)
        turn_value = response.get("turn") if isinstance(response.get("turn"), dict) else {}
        active["turn_id"] = turn_value.get("id") or active["turn_id"]
        persist_active()
        print(json.dumps({"actual_request_started": True, "attempt_id": attempt_id,
                          "thread_id": active["thread_id"], "turn_id": active["turn_id"],
                          "model": MODEL, "effort": EFFORT,
                          "wrapper_hash_match": source_match,
                          "remaining_before": rate_before["remaining_percent_derived"]},
                         ensure_ascii=False), flush=True)

        end = min(process_deadline, active["turn_started_monotonic"] + TURN_TIMEOUT_SECONDS)
        while active["turn_complete"] is None:
            left = end - time.monotonic()
            if left <= 0:
                raise RunStop("turn_completion_timeout", attempt_id)
            try:
                value = messages.get(timeout=min(left, 15))
            except queue.Empty:
                continue
            if value is None:
                raise RunStop("stdio_process_exited_during_turn", attempt_id)
            if "method" in value and "id" in value:
                deny_server_request(value)
                active["client_action_request_rejected"] = True
                active["server_request_sanitized"] = sanitize(value)
                persist_active()
                raise RunStop("server_requested_client_action", attempt_id, value)
            if "method" in value:
                handle(value)

        drain_notifications(0.75)
        final_messages = [message for message in active["agent_messages"]
                          if message.get("phase") in ("final", "final_answer")]
        if not final_messages and active["turn_complete"].get("status") == "completed":
            final_messages = [message for message in active["agent_messages"] if message.get("phase") is None]
        response_text = final_messages[-1].get("text", "") if final_messages else ""
        (directory / "response.txt").write_bytes(response_text.encode("utf-8"))
        write_json(directory / "agent_messages.json", active["agent_messages"])
        write_json(directory / "terminal_snapshot.json", {
            "captured_at_utc": utcnow(),
            "turn_status": active["turn_complete"].get("status") if isinstance(active["turn_complete"], dict) else None,
            "turn_error_sanitized": active["turn_error_sanitized"],
            "error_notifications": active["error_notifications"],
            "json_rpc_errors": active["json_rpc_errors"],
            "response_sha256": text_digest(response_text),
            "response_chars": len(response_text),
        })

        post_turn_stop = None
        rate_after = None
        try:
            after_query = utcnow()
            rate_after = weekly_snapshot(
                rpc("account/rateLimits/read", include_params=False, timeout=15), after_query, utcnow()
            )
            active["rate_after"] = rate_after
            if not rate_after["valid"]:
                post_turn_stop = RunStop("weekly_rate_limit_snapshot_invalid_after_attempt", attempt_id)
        except RunStop as error:
            post_turn_stop = error
            active["rate_after"] = None

        answer = extract_json_field(response_text, "final_answer")
        answer_present = bool(response_text)
        predicted = canonical(answer) if answer_present else None
        target = canonical(item["target"]) if answer_present else None
        error_structures = list(active["error_notifications"]) + list(active["json_rpc_errors"])
        if active["turn_error_sanitized"] is not None:
            error_structures.append({"turn_error": active["turn_error_sanitized"]})
        error_classes = classify_error_structures(error_structures)
        if active["tool_violation"] or active["client_action_request_rejected"]:
            error_classes = sorted(set(error_classes + ["tool_action"]))
        turn_status = active["turn_complete"].get("status") if isinstance(active["turn_complete"], dict) else None
        decision = retry_decision(turn_status, response_text, attempt_number, error_classes)
        elapsed = active.get("turn_completed_monotonic", time.monotonic()) - active["turn_started_monotonic"]
        usage_last = active["token_usage"].get("last", {}) if isinstance(active["token_usage"], dict) else {}
        result = {
            "attempt_id": attempt_id,
            "pass": pass_number,
            "locked_item_index": item_index,
            "item_id": item["item_id"],
            "attempt_number": attempt_number,
            "is_infrastructure_retry": attempt_number == 2,
            "retries_historical_failed_attempt": bool(is_historical_retry),
            "status": "answer_observed" if answer_present else "failed_no_final",
            "turn_status": turn_status,
            "retry_decision": decision,
            "non_retryable_error_classes": error_classes,
            "error_notifications_sanitized": active["error_notifications"],
            "json_rpc_errors_sanitized": active["json_rpc_errors"],
            "turn_error_sanitized": active["turn_error_sanitized"],
            "model": MODEL,
            "effort": EFFORT,
            "thread_id": active["thread_id"],
            "turn_id": active["turn_id"],
            "thread_echo": active["thread_echo"],
            "settings_echo": active["settings_echo"],
            "prompt_sha256": item["prompt_sha256"],
            "prompt_chars": item["prompt_chars"],
            "response_sha256": text_digest(response_text),
            "response_chars": len(response_text),
            "extracted_final_answer": answer if answer_present else None,
            "format_valid_final_answer": bool(answer) if answer_present else None,
            "canonical_prediction": predicted,
            "canonical_target": target,
            "was_correct": predicted == target if answer_present else None,
            "elapsed_seconds": elapsed,
            "token_usage": active["token_usage"],
            "cached_input_tokens": usage_last.get("cachedInputTokens"),
            "rate_before": rate_before,
            "rate_after": rate_after,
            "tool_item_types": sorted({kind for kind in active["item_types"] if kind in TOOL_TYPES}),
            "all_item_types": sorted({kind for kind in active["item_types"] if isinstance(kind, str)}),
            "client_action_request_rejected": active["client_action_request_rejected"],
            "request_log": active["request_log"],
            "post_turn_control_failure": post_turn_stop.record() if post_turn_stop else None,
            "completed_at_utc": utcnow(),
        }
        write_json(directory / "result.json", result)
        attempt_summary = {
            "attempt_id": attempt_id,
            "pass": pass_number,
            "locked_item_index": item_index,
            "item_id": item["item_id"],
            "attempt_number": attempt_number,
            "status": result["status"],
            "turn_status": turn_status,
            "retry_decision": decision,
            "error_classes": error_classes,
            "was_correct": result["was_correct"],
            "remaining_before": rate_before["remaining_percent_derived"],
            "remaining_after": rate_after.get("remaining_percent_derived") if rate_after else None,
            "total_tokens": usage_last.get("totalTokens"),
            "elapsed_seconds": elapsed,
            "result_path": str((directory / "result.json").relative_to(OUT)),
        }
        state["attempts"].append(attempt_summary)
        if answer_present:
            state["new_answer_observations"] += 1
        else:
            state["new_failed_no_final_attempts"] += 1
        persist_state()
        active = None
        print(json.dumps({
            "attempt": attempt_id,
            "item_id": item["item_id"],
            "status": result["status"],
            "turn_status": turn_status,
            "decision": decision,
            "correct": result["was_correct"],
            "tokens": usage_last.get("totalTokens"),
            "elapsed_seconds": round(elapsed, 3),
            "remaining_before": rate_before["remaining_percent_derived"],
            "remaining_after": rate_after.get("remaining_percent_derived") if rate_after else None,
            "error_classes": error_classes,
        }, ensure_ascii=False), flush=True)
        return {"outcome": "attempt", "result": result, "post_turn_stop": post_turn_stop}

    def record_active_failure(error):
        if active is None:
            return
        failure = {
            "status": "stopped_before_normal_attempt_result",
            "attempt_id": active["attempt_id"],
            "pass": active["pass"],
            "locked_item_index": active["locked_item_index"],
            "item_id": active["item_id"],
            "attempt_number": active["attempt_number"],
            "turn_start_sent": active["turn_start_sent"],
            "failure": error.record(),
            "turn_status": active["turn_complete"].get("status") if isinstance(active["turn_complete"], dict) else None,
            "turn_error_sanitized": active["turn_error_sanitized"],
            "error_notifications_sanitized": active["error_notifications"],
            "json_rpc_errors_sanitized": active["json_rpc_errors"],
            "request_log": active["request_log"],
            "token_usage": active["token_usage"],
            "rate_before": active.get("rate_before"),
            "tool_violation": active["tool_violation"],
            "client_action_request_rejected": active["client_action_request_rejected"],
            "failed_at_utc": utcnow(),
        }
        write_json(active["_directory"] / "failure.json", failure)

    try:
        command = [NODE, CLI, "app-server", "--stdio"]
        for feature in DISABLED_FEATURES:
            command.extend(["--disable", feature])
        command.extend(["-c", "features.unified_exec=false"])
        proc = subprocess.Popen(
            command,
            cwd=str(OUT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        def reader():
            try:
                for line in proc.stdout:
                    try:
                        value = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(value, dict):
                        messages.put(value)
            finally:
                messages.put(None)

        threading.Thread(target=reader, daemon=True).start()
        initialized = rpc(
            "initialize",
            {
                "clientInfo": {
                    "name": "meta_harness_e4_retry_resume",
                    "title": "Meta-Harness E4 retry resume",
                    "version": "1.0.0",
                },
                "capabilities": {"experimentalApi": False},
            },
            timeout=10,
        )
        state["cli_user_agent"] = initialized.get("userAgent")
        send_notification("initialized")
        catalog = rpc("model/list", {"limit": 100, "includeHidden": True}, timeout=15)
        matches = [entry for entry in catalog.get("data", []) if isinstance(entry, dict) and (entry.get("id") == MODEL or entry.get("model") == MODEL)]
        efforts = [
            entry.get("reasoningEffort") for entry in matches[0].get("supportedReasoningEfforts", [])
            if isinstance(entry, dict)
        ] if len(matches) == 1 else []
        if len(matches) != 1 or EFFORT not in efforts or catalog.get("nextCursor") is not None:
            raise RunStop("model_or_effort_unavailable", "model/list", {"matches": len(matches), "efforts": efforts})
        state["catalog_evidence"] = {
            "id": matches[0].get("id"),
            "model": matches[0].get("model"),
            "display_name": matches[0].get("displayName"),
            "supported_reasoning_efforts": efforts,
            "catalog_complete": True,
        }
        state["status"] = "running"
        persist_state()

        stop = False
        for position, (pass_number, item_index, first_attempt_number) in enumerate(schedule):
            state["next_schedule_position"] = position
            persist_state()
            attempt_number = first_attempt_number
            final_result = None
            while True:
                outcome = run_attempt(
                    pass_number,
                    item_index,
                    attempt_number,
                    is_historical_retry=False,
                )
                if outcome["outcome"] == "not_sent_quota":
                    state["status"] = "stopped_by_quota_guard"
                    state["stop_reason"] = outcome["stop_reason"]
                    stop = True
                    break
                result = outcome["result"]
                final_result = result
                if outcome["post_turn_stop"] is not None:
                    state["status"] = "stopped_on_control_failure"
                    state["stop_reason"] = outcome["post_turn_stop"].record()
                    stop = True
                    break
                if result["tool_item_types"] or result["client_action_request_rejected"]:
                    state["status"] = "stopped_on_protocol_violation"
                    state["stop_reason"] = {"kind": "tool_or_client_action", "attempt_id": result["attempt_id"]}
                    stop = True
                    break
                decision = result["retry_decision"]
                if decision == "answer_present_no_retry":
                    state["consecutive_final_missing_items"] = 0
                    if result["non_retryable_error_classes"]:
                        state["status"] = "stopped_on_non_retryable_error_signal"
                        state["stop_reason"] = {
                            "kind": "non_retryable_error_signal_with_answer",
                            "attempt_id": result["attempt_id"],
                            "classes": result["non_retryable_error_classes"],
                        }
                        stop = True
                    break
                if decision == "retry_once":
                    attempt_number = 2
                    continue
                if decision == "missing_after_retry_continue":
                    state["items_missing_after_retry"] += 1
                    state["consecutive_final_missing_items"] += 1
                    if state["consecutive_final_missing_items"] >= 2:
                        state["status"] = "stopped_after_two_consecutive_missing_items"
                        state["stop_reason"] = {
                            "kind": "two_consecutive_final_missing_items",
                            "attempt_id": result["attempt_id"],
                        }
                        stop = True
                    break
                state["items_missing_after_retry"] += 1
                state["consecutive_final_missing_items"] += 1
                state["status"] = "stopped_on_non_retryable_or_ineligible_failure"
                state["stop_reason"] = {
                    "kind": decision,
                    "attempt_id": result["attempt_id"],
                    "classes": result["non_retryable_error_classes"],
                    "turn_status": result["turn_status"],
                }
                stop = True
                break

            if final_result is not None:
                state["items_finalized_in_resume"] += 1
                state["item_outcomes"].append({
                    "pass": pass_number,
                    "locked_item_index": item_index,
                    "item_id": locked[item_index]["item_id"],
                    "status": "answer_observed" if final_result["response_chars"] else "missing",
                    "final_attempt_id": final_result["attempt_id"],
                    "was_correct": final_result["was_correct"],
                })
            state["next_schedule_position"] = position + (0 if stop and final_result is None else 1)
            persist_state()
            if stop:
                break
        else:
            state["status"] = "completed_remaining_schedule"
            state["stop_reason"] = {"kind": "schedule_complete"}
        persist_state()
    except RunStop as error:
        record_active_failure(error)
        state["status"] = "stopped_on_failure"
        state["stop_reason"] = error.record()
        persist_state()
    except Exception as error:
        safe = {"kind": "client_exception", "exception_type": type(error).__name__, "message": sanitize(str(error))}
        record_active_failure(RunStop("client_exception", detail=safe))
        state["status"] = "stopped_on_failure"
        state["stop_reason"] = safe
        persist_state()
    finally:
        if proc is not None:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
                cleanup = "this_resume_app_server_exited_after_stdio_closed"
            except subprocess.TimeoutExpired:
                if proc.poll() is None:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=8,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    proc.wait(timeout=5)
                    cleanup = "only_this_resume_app_server_tree_terminated"
            except Exception as error:
                cleanup = "cleanup_error_" + type(error).__name__
        state["helper_cleanup"] = cleanup
        state["finished_at_utc"] = utcnow()
        state["total_formal_attempts_including_history"] = state["historical_attempts_before_resume"] + state["new_turn_start_requests_sent"]
        persist_state()
    print(json.dumps({
        "resume_final_status": state["status"],
        "new_turn_start_requests_sent": state["new_turn_start_requests_sent"],
        "new_answer_observations": state["new_answer_observations"],
        "new_failed_no_final_attempts": state["new_failed_no_final_attempts"],
        "new_infrastructure_retries_sent": state["new_infrastructure_retries_sent"],
        "items_finalized_in_resume": state["items_finalized_in_resume"],
        "items_missing_after_retry": state["items_missing_after_retry"],
        "stop_reason": state["stop_reason"],
        "helper_cleanup": state["helper_cleanup"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selfcheck":
        selfcheck()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("Usage: resume_with_infra_retry.py [--selfcheck]")
