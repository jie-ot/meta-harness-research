"""Read-only OpenRouter credit check before launching a paid stage."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from pilot.state import ROOT, atomic_json, read_json, utc_now


def ensure_solver_credits():
    from dotenv import dotenv_values
    cfg = read_json(ROOT / "pilot_config.json")
    key = dotenv_values((ROOT / cfg["credentials_file"]).resolve()).get("OPENROUTER_API_KEY")
    if not key:
        raise PermissionError("OPENROUTER_API_KEY is missing")
    report = {"checked_at_utc": utc_now(), "queries_are_read_only": True}
    for name in ["key", "credits"]:
        request = urllib.request.Request("https://openrouter.ai/api/v1/" + name, headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response).get("data", {})
                allowed = ["limit", "limit_remaining", "limit_reset", "usage", "usage_daily", "total_credits", "total_usage"]
                report[name] = {"http_status": response.status, "data": {k: data[k] for k in allowed if k in data}}
        except urllib.error.HTTPError as exc:
            report[name] = {"http_status": exc.code}
        except Exception as exc:
            report[name] = {"error_type": type(exc).__name__}
    if report.get("credits", {}).get("http_status") == 200:
        credits = report["credits"]["data"]
        report["available_account_credits_usd"] = credits["total_credits"] - credits["total_usage"]
    atomic_json(ROOT / "external/openrouter_billing_preflight.json", report)
    if report["key"].get("http_status") != 200:
        raise PermissionError("OpenRouter key check failed; no generation requests launched")
    remaining = report["key"].get("data", {}).get("limit_remaining")
    if remaining is not None and remaining <= 0:
        raise RuntimeError("OpenRouter key credit limit is exhausted; no generation requests launched")
    available = report.get("available_account_credits_usd")
    if available is not None and available <= 0:
        raise RuntimeError(f"OpenRouter available credits are {available:.6f} USD; add credits or replace the local key before resuming")
    return report
