"""Analysis fixed before the held-out D50 run; never selects using test scores."""
from __future__ import annotations

import csv
import itertools
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean

from pilot.state import ROOT, atomic_json, digest, object_hash, read_json, read_jsonl, utc_now, write_once

BASE = "confusion_disambiguation_memory"
CORE_RUNS = ["D0_a", "D0_b", "D100_a", "D100_b"]
ANALYSIS = ROOT / "analysis"


def save_csv(name, rows):
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ANALYSIS / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze_noise():
    labels = ["R4B", "R7A", "R12B"]
    rows, score_panels, score_totals = [], {}, {}
    for label in labels:
        reps = []
        for repetition in [1, 2]:
            base = ROOT / "external/noise" / label / str(repetition)
            score, audit = [read_json(base / phase / "result.json") for phase in ["score", "audit"]]
            score_traces = read_jsonl(base / "score/score_traces.jsonl")
            audit_traces = read_jsonl(base / "audit/audit_traces.jsonl")
            for result, traces in [(score, score_traces), (audit, audit_traces)]:
                if len(traces) != 100 or sum(x["was_correct"] for x in traces) != result["correct"]:
                    raise ValueError("Frozen traces do not reconstruct the aggregate")
                if result["cache_mode"] != "off" or result["usage"]["cache_hits"]:
                    raise ValueError("Frozen evaluation used a local response cache")
            rows.append({"version": label, "repetition": repetition, "score_pp": score["correct"], "audit_pp": audit["correct"],
                         "cost_usd": score["estimated_cost_usd"] + audit["estimated_cost_usd"],
                         "runtime_seconds": score.get("runtime_seconds", score["runtime_seconds_this_attempt"]) + audit.get("runtime_seconds", audit["runtime_seconds_this_attempt"]),
                         "api_requests": score["usage"]["api_requests"] + audit["usage"]["api_requests"]})
            reps.append({"audit": audit, "trace": {r["item_id"]: r["was_correct"] for r in audit_traces}})
            if repetition == 1:
                score_panels[label] = {r["item_id"]: r["was_correct"] for r in score_traces}
                score_totals[label] = score["correct"]
        improved = sum(not a and reps[1]["trace"][item] for item, a in reps[0]["trace"].items())
        regressed = sum(a and not reps[1]["trace"][item] for item, a in reps[0]["trace"].items())
        for row in rows[-2:]:
            row.update({"audit_repeat2_minus_repeat1_pp": reps[1]["audit"]["correct"] - reps[0]["audit"]["correct"],
                        "wrong_to_correct": improved, "correct_to_wrong": regressed})
    J = max(abs(r["audit_repeat2_minus_repeat1_pp"]) for r in rows)
    s0 = mean(r["audit_pp"] for r in rows if r["version"] == "R4B")
    reference = {"S0_pp": s0, "J_pp": J, "delta_pp": max(3, J), "frozen_prediction_tasks": 1200,
                 "created_at_utc": utc_now(), "input_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in (ROOT / "external/noise").rglob("result.json")}}
    old = ANALYSIS / "noise_reference.json"
    if old.exists():
        if read_json(old)["input_hashes"] != reference["input_hashes"]:
            raise ValueError("Frozen measurement changed")
        reference = read_json(old)
    else:
        write_once(old, reference)
    panel_ids = list(score_panels[labels[0]])
    if any(list(score_panels[label]) != panel_ids for label in labels):
        raise ValueError("Frozen score panels differ")
    stability = []
    for size in [20, 50]:
        rng = random.Random(20261003)
        inversions = ties = changed = 0
        pairs = list(itertools.combinations(labels, 2))
        for _ in range(1000):
            indices = rng.sample(panel_ids, size)
            sampled = {label: sum(score_panels[label][item] for item in indices) for label in labels}
            differences = [(sampled[a] - sampled[b], score_totals[a] - score_totals[b]) for a, b in pairs]
            inversions += any(a * b < 0 for a, b in differences)
            ties += any(a == 0 for a, b in differences)
            changed += any((a > 0) - (a < 0) != (b > 0) - (b < 0) for a, b in differences)
        stability.append({"sample_size": size, "repetitions": 1000, "seed": 20261003,
                          "any_pair_inversion_rate": inversions / 1000, "any_tie_rate": ties / 1000,
                          "any_pair_order_or_tie_change_rate": changed / 1000, "reference": "first repetition, full 100 score items"})
    save_csv("noise_summary.csv", rows)
    save_csv("panel_stability.csv", stability)
    return reference


def checkpoint_path(run, T):
    return ROOT / "external/control" / run / f"round{T}_checkpoint.json"


def selected_score(checkpoint, source_hashes=None):
    name = checkpoint["selected"]
    if name == BASE:
        path = ANALYSIS / "noise_reference.json"
        value = read_json(path)["S0_pp"]
    else:
        path = ROOT / "external/audit" / checkpoint["run_id"] / name / "result.json"
        result = read_json(path)
        if result["total"] != 100:
            raise ValueError("Missing complete candidate audit")
        value = 100 * result["accuracy"]
    if source_hashes is not None:
        source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    return value


def checkpoint_rows(runs=CORE_RUNS, rounds=(1, 2)):
    rows = []
    for run in runs:
        for T in rounds:
            path = checkpoint_path(run, T)
            checkpoint = read_json(path)
            for rel, expected in checkpoint["artifacts"].items():
                if digest(ROOT / rel) != expected:
                    raise ValueError("Sealed checkpoint inputs changed")
            latest = checkpoint["candidates"][1 + (T - 1) * 2:1 + T * 2]
            offspring_scores = []
            for row in latest:
                if row["status"] == "valid":
                    result = read_json(ROOT / "external/audit" / run / row["candidate"] / "result.json")
                    offspring_scores.append(result["accuracy"] * 100)
            selected = next(r for r in checkpoint["candidates"] if r["candidate"] == checkpoint["selected"])
            rows.append({"run_id": run, "T": T, "N": 2 * T, "D": checkpoint["D"], "selected": checkpoint["selected"],
                         "selected_score_pp": selected["correct"], "S_pp": selected_score(checkpoint), "C_usd": checkpoint["cost"]["cost_usd"],
                         "offspring_mean_audit_pp": mean(offspring_scores) if len(offspring_scores) == 2 else None,
                         "valid_offspring": len(offspring_scores), "unknown_failed_costs": checkpoint["cost"]["unknown_failed_attempts"]})
    return rows


def freeze_prediction():
    path = ANALYSIS / "prediction_before_run.json"
    if path.exists():
        return verify_prediction()
    if (ROOT / "external/control/D50_c/initialized.json").exists():
        raise ValueError("D50 already initialized before prediction")
    rows = checkpoint_rows(rounds=(2,))
    sources = {}
    for run in CORE_RUNS:
        cp = checkpoint_path(run, 2)
        sources[cp.relative_to(ROOT).as_posix()] = digest(cp)
        selected_score(read_json(cp), sources)
    lo, hi = [[r for r in rows if r["D"] == d] for d in (0, 100)]
    s_lo, s_hi = [mean(r["S_pp"] for r in group) for group in [lo, hi]]
    c_lo, c_hi = [mean(r["C_usd"] for r in group) for group in [lo, hi]]
    noise = read_json(ANALYSIS / "noise_reference.json")
    result = {"schema_version": 3, "created_at_utc": utc_now(), "source_hashes": sources,
              "method": "arithmetic midpoint of two-repetition endpoint means at T=2",
              "D50_S_prediction_pp": (s_lo + s_hi) / 2, "D50_C_prediction_usd": (c_lo + c_hi) / 2,
              "ignore_D_prediction_pp": s_lo, "ignore_evolution_prediction_pp": noise["S0_pp"],
              "endpoint_D0_mean_S_pp": s_lo, "endpoint_D100_mean_S_pp": s_hi,
              "endpoint_D0_mean_C_usd": c_lo, "endpoint_D100_mean_C_usd": c_hi,
              "delta_pp": noise["delta_pp"], "max_prediction_error_pp": 5, "required_baseline_improvement_pp": 1}
    write_once(path, result)
    write_once(ANALYSIS / "prediction_seal.json", {"prediction_sha256": digest(path), "sealed_at_utc": utc_now()})
    return result


def verify_prediction():
    path = ANALYSIS / "prediction_before_run.json"
    seal = read_json(ANALYSIS / "prediction_seal.json")
    if digest(path) != seal["prediction_sha256"]:
        raise ValueError("Pre-run prediction changed")
    result = read_json(path)
    for rel, expected in result["source_hashes"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Prediction source changed")
    return result


def signal(differences, delta):
    if len(differences) != 2 or any(d is None for d in differences):
        return "incomplete"
    if differences[0] * differences[1] > 0 and abs(mean(differences)) >= delta:
        return "positive" if mean(differences) > 0 else "negative"
    return "unresolved"


def effects(rows, delta):
    lookup = {(r["run_id"], r["T"]): r for r in rows}
    out = []
    for d in [0, 100]:
        diffs = [lookup[(f"D{d}_{rep}", 2)]["S_pp"] - lookup[(f"D{d}_{rep}", 1)]["S_pp"] for rep in ["a", "b"]]
        costs = [lookup[(f"D{d}_{rep}", 2)]["C_usd"] - lookup[(f"D{d}_{rep}", 1)]["C_usd"] for rep in ["a", "b"]]
        out.append({"axis": "T", "fixed": f"D={d}", "a_delta_pp": diffs[0], "b_delta_pp": diffs[1], "mean_delta_pp": mean(diffs), "mean_extra_cost_usd": mean(costs), "signal": signal(diffs, delta)})
    for t in [1, 2]:
        diffs = [lookup[(f"D100_{rep}", t)]["S_pp"] - lookup[(f"D0_{rep}", t)]["S_pp"] for rep in ["a", "b"]]
        costs = [lookup[(f"D100_{rep}", t)]["C_usd"] - lookup[(f"D0_{rep}", t)]["C_usd"] for rep in ["a", "b"]]
        offspring = []
        for rep in ["a", "b"]:
            hi, lo = lookup[(f"D100_{rep}", t)]["offspring_mean_audit_pp"], lookup[(f"D0_{rep}", t)]["offspring_mean_audit_pp"]
            offspring.append(hi - lo if hi is not None and lo is not None else None)
        out.append({"axis": "D", "fixed": f"T={t}", "a_delta_pp": diffs[0], "b_delta_pp": diffs[1], "mean_delta_pp": mean(diffs), "mean_extra_cost_usd": mean(costs), "signal": signal(diffs, delta),
                    "offspring_a_delta_pp": offspring[0], "offspring_b_delta_pp": offspring[1], "offspring_signal": signal(offspring, delta)})
    return out


def paired_changes(rows):
    """Compare fixed selected versions; two H0 measurements retain their identity."""
    lookup = {(r["run_id"], r["T"]): r for r in rows}
    def panels(row):
        if row["selected"] == BASE:
            paths = [(f"R4B_repeat{rep}", ROOT / f"external/noise/R4B/{rep}/audit/audit_traces.jsonl") for rep in [1, 2]]
        else:
            paths = [(row["run_id"] + "/" + row["selected"], ROOT / "external/audit" / row["run_id"] / row["selected"] / "audit_traces.jsonl")]
        result = []
        for label, path in paths:
            records = read_jsonl(path)
            mapping = {r["item_id"]: bool(r["was_correct"]) for r in records}
            if len(records) != len(mapping) or len(mapping) != 100:
                raise ValueError("Paired comparison requires all 100 distinct test items")
            result.append((label, mapping))
        return result
    comparisons = []
    for run in dict.fromkeys(r["run_id"] for r in rows):
        comparisons.append(("T", run, lookup[(run, 1)], lookup[(run, 2)]))
    for rep, T in itertools.product(["a", "b"], [1, 2]):
        comparisons.append(("D", f"repeat={rep},T={T}", lookup[(f"D0_{rep}", T)], lookup[(f"D100_{rep}", T)]))
    output = []
    for axis, label, before, after in comparisons:
        left, right = panels(before), panels(after)
        pairs = zip(left, right) if before["selected"] == after["selected"] == BASE else itertools.product(left, right)
        for (left_label, a), (right_label, b) in pairs:
            if set(a) != set(b): raise ValueError("Test item panels differ")
            gain = sum(not a[i] and b[i] for i in a)
            loss = sum(a[i] and not b[i] for i in a)
            output.append({"axis": axis, "comparison": label, "before": left_label, "after": right_label,
                           "wrong_to_correct": gain, "correct_to_wrong": loss, "net_correct": gain - loss,
                           "note": "H0 comparisons list both frozen repetitions; average their net changes for the S0-based difference"})
    return output


def analyze_all(include_holdout=True):
    noise = analyze_noise()
    prediction = verify_prediction() if include_holdout else None
    active_runs = CORE_RUNS + (["D50_c"] if include_holdout else [])
    expected_candidates = 20 if include_holdout else 16
    expected_sessions = 10 if include_holdout else 8
    rows = checkpoint_rows(active_runs)
    contrast = effects([r for r in rows if r["D"] != 50], noise["delta_pp"])
    grid = []
    for t, d in itertools.product([1, 2], [0, 100]):
        group = [r for r in rows if r["T"] == t and r["D"] == d]
        grid.append({"T": t, "N": 2 * t, "D": d, "repetitions": 2, "mean_S_pp": mean(r["S_pp"] for r in group), "mean_C_usd": mean(r["C_usd"] for r in group)})
    for row in grid:
        row["nondominated_among_measured_means"] = not any(other["mean_C_usd"] <= row["mean_C_usd"] and other["mean_S_pp"] >= row["mean_S_pp"] and (other["mean_C_usd"] < row["mean_C_usd"] or other["mean_S_pp"] > row["mean_S_pp"]) for other in grid)
    checks = []
    positive_prediction = False
    prediction_status = "not_run_by_user_request"
    holdout_report = {"prediction_status": prediction_status}
    if include_holdout:
        holdout = next(r for r in rows if r["D"] == 50 and r["T"] == 2)
        errors = {"midpoint": abs(holdout["S_pp"] - prediction["D50_S_prediction_pp"]), "ignore_D": abs(holdout["S_pp"] - prediction["ignore_D_prediction_pp"]), "ignore_evolution": abs(holdout["S_pp"] - prediction["ignore_evolution_prediction_pp"])}
        endpoint_signal = next(r["signal"] for r in contrast if r["axis"] == "D" and r["fixed"] == "T=2") in {"positive", "negative"}
        positive_prediction = endpoint_signal and errors["midpoint"] <= 5 and all(errors[b] - errors["midpoint"] >= 1 - 1e-9 for b in ["ignore_D", "ignore_evolution"])
        prediction_status = "positive_initial_evidence" if positive_prediction else ("flat_region_close_prediction" if not endpoint_signal else "not_better_than_baselines" if errors["midpoint"] <= 5 else "prediction_deviation")
        for key, prediction_key in [("midpoint", "D50_S_prediction_pp"), ("ignore_D", "ignore_D_prediction_pp"), ("ignore_evolution", "ignore_evolution_prediction_pp")]:
            checks.append({"method": key, "predicted_S_pp": prediction[prediction_key], "actual_S_pp": holdout["S_pp"], "absolute_error_pp": errors[key]})
        cost_error = abs(holdout["C_usd"] - prediction["D50_C_prediction_usd"])
        holdout_report = {**holdout, "prediction_status": prediction_status, "errors_pp": errors, "predicted_C_usd": prediction["D50_C_prediction_usd"], "absolute_cost_error_usd": cost_error,
                          "relative_cost_error": cost_error / holdout["C_usd"] if holdout["C_usd"] else None}
    candidates, usage_rows = [], []
    for run in active_runs:
        for t in [1, 2]:
            pending = read_json(ROOT / "external/control" / run / f"proposer_round{t}/pending_eval.json")
            for candidate in pending["candidates"]:
                name = candidate["name"]
                base = ROOT / "runs" / run / "text_classification/history" / name
                invalid = ROOT / "external/control" / run / "invalid" / f"{name}.json"
                row = {"run_id": run, "T": t, "D": read_json(ROOT / "pilot_config.json")["runs"][run], "slot": candidate["slot"], "candidate": name, "valid": not invalid.exists()}
                if not invalid.exists():
                    score, audit = read_json(base / "score/result.json"), read_json(ROOT / "external/audit" / run / name / "result.json")
                    row.update({"score_pp": score["accuracy"] * 100, "audit_pp": audit["accuracy"] * 100})
                    traces = read_jsonl(base / "score/score_traces.jsonl")
                    row["legacy_score_correct"] = sum(r["was_correct"] for r in traces if r["legacy"])
                    row["new_score_correct"] = sum(r["was_correct"] for r in traces if not r["legacy"])
                candidates.append(row)
        for path in (ROOT / "runs" / run / "text_classification/history").rglob("result.json"):
            r = read_json(path)
            if "usage" in r:
                usage_rows.append({"run_id": run, "candidate": r["candidate"], "round": r["round"], "D": r["D"], "phase": r["phase"], "total_items": r["total"], "wall_seconds": r.get("runtime_seconds"), **r["usage"]})
    for parent in [ROOT / "external/audit", ROOT / "external/noise"]:
        for path in parent.rglob("result.json"):
            r = read_json(path)
            if r["run_id"].startswith("D100_injected_"):
                continue
            usage_rows.append({"run_id": r["run_id"], "candidate": r["candidate"], "round": r["round"], "D": r["D"], "phase": r["phase"], "total_items": r["total"], "wall_seconds": r.get("runtime_seconds"), **r["usage"]})
    # Keep the original cohort separate from the independently authorized supplement.
    ledger = [r for r in read_jsonl(ROOT / "external/cost_ledger.jsonl")
              if not r.get("run_id", "").startswith("D100_injected_")]
    for r in ledger:
        if r["event"] == "proposer_result":
            usage_rows.append({"run_id": r["run_id"], "candidate": None, "round": r["round"], "D": r["D"], "phase": "proposer", "cost_usd": r["cost"], "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"), "wall_seconds": r["wall_seconds"], "cost_source": r["cost_source"], "logical_calls": r.get("logical_calls"), "api_requests": r.get("api_requests"), "cache_hits": r.get("cache_hits"), "failed_api_requests": r.get("failed_api_requests"), "provider_usage": r.get("provider_usage")})
    total = sum(r.get("cost") or 0 for r in ledger if r["event"] in {"api_result", "proposer_result"})
    breakdown = {}
    for event in ledger:
        if event["event"] not in {"api_result", "proposer_result"}: continue
        category = "frozen_measurement" if event.get("run_id", "").startswith("noise_") else event["phase"]
        breakdown[category] = breakdown.get(category, 0.0) + (event.get("cost") or 0)
    if len(candidates) != expected_candidates:
        raise ValueError("Nominal candidate count differs from the fixed design")
    invalid_count = sum(not row["valid"] for row in candidates)
    failed_costs = sum(event["event"] in {"api_failed", "proposer_failed"} and event.get("http_status") != 402 for event in ledger)
    credit_rejections = sum(event["event"] == "api_failed" and event.get("http_status") == 402 for event in ledger)
    has_T = any(r["axis"] == "T" and r["signal"] in {"positive", "negative"} for r in contrast)
    has_D = any(r["axis"] == "D" and r["signal"] in {"positive", "negative"} for r in contrast)
    child_only = any(r.get("offspring_signal") == "positive" and r["signal"] != "positive" for r in contrast)
    if invalid_count:
        conclusion = "存在无效候选，须结合失败记录解释本次不完整执行，不能据此宣称二维规律成立。"
    elif has_T and has_D and not include_holdout:
        conclusion = "T、D均有可辨作用，有继续研究资源配比的初步依据；实验4暂未执行，尚不能判断未测配比能否事前预测。"
    elif has_T and has_D and positive_prediction:
        conclusion = "T、D均有可辨作用，D50事前预测优于两条基线，有继续扩大二维性能—成本研究的初步证据。"
    elif has_T and has_D:
        conclusion = "T、D均有可辨作用，资源配比问题值得继续研究，但当前中间值预测未通过事前标准。"
    elif child_only:
        conclusion = "子代质量有改善信号，但交付质量未同步改善，应检查评分选择与其稳定性。"
    elif has_T or has_D:
        conclusion = f"只有{'T' if has_T else 'D'}的作用达到预设筛查门槛，宜先扩展有信号的轴，尚不能宣称二维规律成立。"
    else:
        conclusion = "T、D均未呈现超过预设波动门槛的稳定作用，当前预实验支持不足，不能用四格数据强行拟合 scaling law。"
    result = {"conclusion": conclusion, "noise": noise, "effects": contrast, "prediction": holdout_report,
              "nominal_new_candidates": expected_candidates, "valid_new_candidates": expected_candidates - invalid_count, "proposer_sessions": expected_sessions,
              "completed_experiments": [1,2,3,4] if include_holdout else [1,2,3], "deferred_experiments": [] if include_holdout else [4],
              "accounted_total_usd": total, "cost_by_phase_usd": breakdown, "unreported_failed_cost_count": failed_costs,
              "nonbillable_credit_rejections": credit_rejections,
              "cost_basis": "provider reports and labeled log estimates, excluding historical shared memory cost; not a cash invoice",
              "isolation": "native tool restrictions, per-tool allowlist, separate directories; not OS-level isolation"}
    save_csv("checkpoint_summary.csv", rows)
    save_csv("grid_summary.csv", grid)
    save_csv("effects.csv", contrast)
    save_csv("all_candidates.csv", candidates)
    save_csv("round1_candidates.csv", [r for r in candidates if r["T"] == 1 and r["D"] != 50])
    save_csv("round1_comparison.csv", [r for r in rows if r["T"] == 1 and r["D"] != 50])
    save_csv("stage_usage.csv", usage_rows)
    if include_holdout:
        save_csv("prediction_check.csv", checks)
    save_csv("paired_changes.csv", paired_changes(rows))
    if include_holdout:
        atomic_json(ANALYSIS / "holdout_summary.json", holdout_report)
    atomic_json(ANALYSIS / "analysis.json", result)
    report = [conclusion, f"冻结重评的波动参照 J={noise['J_pp']:.2f} 个百分点，筛查门槛 δ={noise['delta_pp']:.2f} 个百分点；R4B参考测试成绩 S0={noise['S0_pp']:.2f}%。本次固定 k=2，因此 N=2T，不能独立识别 T 与 N 的作用。"]
    for effect in contrast:
        report.append(f"{effect['axis']} 对照（{effect['fixed']}）两次交付成绩变化为 {effect['a_delta_pp']:+.2f}、{effect['b_delta_pp']:+.2f} 个百分点，均值 {effect['mean_delta_pp']:+.2f}，增加费用均值 ${effect['mean_extra_cost_usd']:.6f}；筛查结果为 {effect['signal']}。")
    if include_holdout:
        report.append(f"D50两轮事前预测 {prediction['D50_S_prediction_pp']:.2f}%，实际交付 {holdout['S_pp']:.2f}%；中间值、忽略D、忽略进化的绝对误差分别为 {errors['midpoint']:.2f}、{errors['ignore_D']:.2f}、{errors['ignore_evolution']:.2f} 个百分点。预测筛查结果：{prediction_status}。预测费用 ${prediction['D50_C_prediction_usd']:.6f}，实际算法成本 ${holdout['C_usd']:.6f}。")
    else:
        report.append("按用户要求，本次只执行实验1、2、3。实验4和D50运行尚未执行，未检验未测配比的预测能力，也不能把未执行写成预测失败。")
    report.append(f"名义新候选{expected_candidates}个，有效{expected_candidates-invalid_count}个，优化器会话{expected_sessions}次。计入冻结重评、测试与搜索的可记录总费用为 ${total:.6f}；另有{failed_costs}次失败请求未返回账单金额，{credit_rejections}次HTTP402额度检查拒绝单列保留。该金额是服务商报告及日志估值，不是现金账单。算法成本C与冻结测量、测试费用分别报告，起点旧memory的历史成本另列。")
    report.append("本次只考察R4B之后的两轮追加进化、一个任务和少量独立重复。筛查门槛不是统计显著性检验，不能据此证明长期幂律、最优停止点或跨任务泛化。逐项数字见 checkpoint_summary.csv、effects.csv、all_candidates.csv 和 paired_changes.csv，原始记录与成本台账均保留。")
    (ANALYSIS / "给导师的结论.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    plot_python = read_json(ROOT / "pilot_config.json").get("plot_python") or shutil.which("python") or sys.executable
    subprocess.run([plot_python, str(ROOT / "pilot/plot_results.py"), str(ANALYSIS)], check=True)
    return result
