"""Render the completed experiment 2/3 flow, using validation scores only.

Reads sealed local artifacts. Does not import an LLM client or run experiments.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"
STEM = "实验2-3_验证集流程图"
RUNS = ["D0_a", "D0_b", "D100_a", "D100_b"]
W, H = 2400, 1550
REGULAR = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
BOLD = FontProperties(fname="C:/Windows/Fonts/msyhbd.ttc")
INK, MUTED = "#152C46", "#5B6C80"
BLUE, ORANGE, GREEN = "#2864AE", "#AC5B13", "#157547"
EDGE, GRAY = "#D5DFE9", "#93A4B5"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    with (OUT / "all_candidates.csv").open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    labels = {(r["run_id"], r["candidate"]): r["run_id"].replace("_", "-") + "-" +
              ("AB" if r["T"] == "1" else "CD")["AB".index(r["slot"])] for r in raw}
    access_path = OUT / "optimizer_evidence_access.csv"
    with access_path.open(encoding="utf-8-sig", newline="") as handle:
        access = list(csv.DictReader(handle))
    paths = [OUT / "all_candidates.csv", access_path]
    runs = []
    for run in RUNS:
        control = ROOT / "external/control" / run
        cp_paths = [control / f"round{t}_checkpoint.json" for t in [1, 2]]
        cps = [read_json(p) for p in cp_paths]
        paths.extend(cp_paths)
        candidates = []
        for t in [1, 2]:
            path = control / f"proposer_round{t}/pending_eval.json"
            paths.append(path)
            pending = read_json(path)
            for item in pending["candidates"]:
                source = next(r for r in raw if r["run_id"] == run and r["candidate"] == item["name"])
                label = labels[(run, item["name"])]
                parent = labels.get((run, item["base_system"]), "H0")
                score = int(float(source["score_pp"]))
                candidates.append({"name": label, "source_name": item["name"],
                                   "experiment": t + 1, "validation": score,
                                   "declared_base": parent})
        selections = []
        for cp in cps:
            best = max(r["correct"] for r in cp["candidates"] if r["status"] == "valid")
            selected = next(r for r in cp["candidates"] if r["candidate"] == cp["selected"])
            assert selected["correct"] == best
            selections.append({"name": labels.get((run, cp["selected"]), "H0"), "validation": best})
        assert len(candidates) == 4
        sealed_scores = {r["candidate"]: r["correct"] for r in cps[1]["candidates"]}
        assert all(c["validation"] == sealed_scores[c["source_name"]] for c in candidates)
        assert all(r["declared_base"] == selections[0]["name"] for r in candidates[2:])
        assert all(r["declared_base"] == "H0" for r in candidates[:2])
        feedback_counts = []
        for candidate in cps[0]["candidates"]:
            path = ROOT / "runs" / run / "text_classification/history" / candidate["candidate"] / "feedback/result.json"
            paths.append(path)
            feedback_counts.append(read_json(path)["total"])
        assert feedback_counts == [cps[0]["D"]] * 3
        visible = sum(int(r["complete_diagnostic_rows_returned"]) for r in access
                      if r["run_id"] == run and r["T"] == "2" and r["phase"] == "feedback")
        assert visible == {"D0_a": 0, "D0_b": 0, "D100_a": 20, "D100_b": 59}[run]
        runs.append({"run": run.replace("_", "-"), "D": cps[0]["D"],
                     "candidates": candidates, "selections": selections,
                     "feedback_records_before_experiment3": sum(feedback_counts),
                     "complete_feedback_records_returned_for_experiment3": visible})
    return runs, paths


def render():
    runs, sources = load()
    fig = plt.figure(figsize=(24, 15.5), facecolor="#F7F9FC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    texts = []

    def box(x, y, w, h, fill="white", stroke=EDGE, radius=14, lw=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor=fill, edgecolor=stroke, linewidth=lw))

    def text(x, y, value, size=20, color=INK, bold=False, ha="left", va="top"):
        item = ax.text(x, y, value, fontsize=size * .72,
                       fontproperties=BOLD if bold else REGULAR,
                       color=color, ha=ha, va=va, linespacing=1.45)
        texts.append(item)
        return item

    def arrow(x1, y1, x2, y2, color=GRAY, lw=1.3):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=12, linewidth=lw, color=color,
                     shrinkA=2, shrinkB=3, zorder=2))

    def score_row(x, y, width, label, score, selected):
        if selected:
            box(x, y, width, 34, fill="#E8F6EE", stroke="#E8F6EE", radius=7)
        text(x + 10, y + 5, label, size=21, color=GREEN if selected else INK, bold=selected)
        text(x + width - 10, y + 3, f"{score}%", size=24,
             color=GREEN if selected else INK, bold=True, ha="right")

    text(42, 35, "实验2 → 实验3：四分支进化流程", size=44, bold=True)
    text(45, 99, "共同起点 H0（R4B） · 分支间不交换代码与日志 · 所有百分数均为固定100题验证准确率", size=22, color=MUTED)
    box(1928, 38, 425, 46, fill="#E8F6EE", stroke="#E8F6EE", radius=23)
    text(2140, 61, "绿色高亮 = 按验证集选中的版本", size=19, color=GREEN, bold=True, ha="center", va="center")

    box(375, 155, 680, 77, fill="#213B5B", stroke="#213B5B")
    text(400, 167, "实验2｜生成 A、B", size=28, color="white", bold=True)
    text(402, 205, "每分支从 H0、A、B 中选优", size=18, color="#DCE8F5")
    box(1095, 155, 1260, 77, fill="#213B5B", stroke="#213B5B")
    text(1120, 167, "实验3｜沿原分支继续，生成 C、D", size=28, color="white", bold=True)
    text(1122, 205, "每分支从 H0、A、B、C、D 中选优；旧版本继续参与比较", size=18, color="#DCE8F5")
    text(220, 255, "独立分支", size=18, color=MUTED, bold=True)
    text(675, 251, "每个新候选：从空 memory 训练200题 → 验证100题", size=20, color=MUTED)

    row_y = [292, 508, 724, 940]
    centers = [y + 91 for y in row_y]
    ax.plot([195, 195], [centers[0], centers[-1]], color="#AAB8C7", lw=1.3, zorder=1)
    arrow(165, 707, 195, 707)
    box(34, 649, 132, 116, fill="#EAF0F7", stroke="#BFCDDE")
    text(100, 664, "H0", size=37, bold=True, ha="center")
    text(100, 718, "共同起点", size=21, ha="center")
    text(100, 784, "同一代码\n同一已有 memory", size=17, color=MUTED, ha="center")

    for data, y in zip(runs, row_y):
        is_extra = data["D"] == 100
        color = ORANGE if is_extra else BLUE
        pale = "#FFF4E7" if is_extra else "#EDF4FF"
        chosen2, chosen3 = data["selections"]
        improved = chosen3["validation"] > chosen2["validation"]
        box(211, y - 11, 2150, 204, fill=pale, stroke=pale, radius=20)
        arrow(195, y + 91, 220, y + 91)
        box(220, y + 58, 132, 66, fill=color, stroke=color, radius=12)
        text(286, y + 91, data["run"], size=23, color="white", bold=True, ha="center", va="center")
        arrow(353, y + 91, 383, y + 91)

        box(385, y, 250, 182)
        text(406, y + 18, "准备改代码的材料", size=19, bold=True)
        text(406, y + 59, "H0已有训练日志", size=19, color=MUTED)
        text(406, y + 92, "H0验证100题的日志", size=19, color=MUTED)
        text(406, y + 136, f"H0额外反馈：{data['D']}题", size=20, color=color, bold=True)
        arrow(637, y + 91, 673, y + 91)

        box(675, y, 380, 182)
        text(695, y + 14, "生成并评测 A、B", size=21, bold=True)
        for candidate, offset in zip(data["candidates"][:2], [49, 89]):
            score_row(689, y + offset, 352, candidate["name"], candidate["validation"], candidate["name"] == chosen2["name"])
        box(690, y + 137, 350, 32, fill="#F0F8F3", stroke="#D4EBDD", radius=7)
        text(865, y + 153, f"实验2选中 {chosen2['name']}  ·  {chosen2['validation']}%",
             size=18, color=GREEN, bold=True, ha="center", va="center")
        arrow(1057, y + 91, 1093, y + 91)

        box(1095, y, 375, 182)
        text(1115, y + 16, "本分支全部历史继续可读", size=21, bold=True)
        text(1115, y + 55, "保留 H0、A、B 的代码与日志", size=19, color=MUTED)
        if is_extra:
            text(1115, y + 97, "A、B各补100题额外反馈", size=21, color=color, bold=True)
            text(1115, y + 140, "新增200条；含H0累计300条", size=19, color=MUTED)
        else:
            text(1115, y + 97, "新增额外反馈：0题", size=21, color=color, bold=True)
            text(1115, y + 140, "使用已有训练与验证日志", size=19, color=MUTED)
        arrow(1472, y + 91, 1513, y + 91)

        box(1515, y, 380, 182)
        text(1535, y + 14, "生成并评测 C、D", size=21, bold=True)
        text(1535, y + 48, f"改动起点：{data['candidates'][2]['declared_base']}", size=18, color=MUTED)
        for candidate, offset in zip(data["candidates"][2:], [85, 127]):
            score_row(1529, y + offset, 352, candidate["name"], candidate["validation"], candidate["name"] == chosen3["name"])
        arrow(1897, y + 91, 1938, y + 91, color=GREEN if improved else GRAY)

        box(1940, y, 410, 182, fill="#F1FBF5" if improved else "white",
            stroke="#6ABB8A" if improved else EDGE, lw=1.8 if improved else 1)
        text(1962, y + 16, "实验3最终选中", size=21, bold=True)
        text(1962, y + 67, chosen3["name"], size=29, color=GREEN, bold=True)
        text(2326, y + 58, f"{chosen3['validation']}%", size=42, color=GREEN, bold=True, ha="right")
        text(1962, y + 132,
             f"替换为新候选 · 提高{chosen3['validation']-chosen2['validation']}个百分点" if improved else "继续保留实验2选中的版本",
             size=20, color=GREEN if improved else MUTED, bold=improved)

    text(375, 1164, "把 a、b 合并，只看同一种 D 下的验证最高分", size=25, bold=True)
    maxima = []
    for D, x, width, color, pale in [(0, 375, 960, BLUE, "#EDF4FF"), (100, 1370, 980, ORANGE, "#FFF4E7")]:
        group = [r for r in runs if r["D"] == D]
        best2 = max(r["selections"][0]["validation"] for r in group)
        best3 = max(r["selections"][1]["validation"] for r in group)
        names = [r["selections"][1]["name"] for r in group if r["selections"][1]["validation"] == best3]
        maxima.append({"D": D, "experiment2_max_validation": best2, "experiment3_max_validation": best3, "new_candidate_counts": [4, 8], "winners": names})
        box(x, 1211, width, 151, fill=pale, stroke=pale)
        text(x + 25, 1228, f"D{D} · 合并两条分支", size=22, color=color, bold=True)
        text(x + 25, 1270, f"实验2  {best2}%   →   实验3  {best3}%", size=31, color=color, bold=True)
        text(x + 25, 1322, "候选累计 4 → 8；最高分未刷新" if best3 == best2 else "候选累计 4 → 8；验证最高分提高", size=19, color=MUTED)

    text(45, 1400, "反馈计数：D100的A、B做同一批100道反馈题，新增200条执行记录；含H0旧记录共300条，独立题目仍为100道。", size=18, color=MUTED)
    text(45, 1433, "采集时机：额外反馈在生成下一批候选前准备；实验3完成后不再为C、D补反馈。H0直接复用已有memory。", size=18, color=MUTED)
    text(45, 1466, "读取情况：额外反馈是可供读取的日志。实验3可确认读取的完整记录为D100-a 20条、D100-b 59条；D100-b在实验2读取失败。", size=18, color=MUTED)
    text(45, 1502, "结论：实验3仅改善D100-a分支（38% → 43%）；D0与D100合并后的验证最高分均未刷新。", size=20, color=INK, bold=True)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundaries = fig.get_window_extent(renderer)
    out_of_bounds = []
    for item in texts:
        b = item.get_window_extent(renderer)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > boundaries.width or b.y1 > boundaries.height:
            out_of_bounds.append(item.get_text())
    assert not out_of_bounds, out_of_bounds
    assert not any("测试" in item.get_text() for item in texts)
    assert not any("第一轮" in item.get_text() or "第二轮" in item.get_text() for item in texts)
    paths = []
    for ext in ["png", "svg"]:
        target = OUT / f"{STEM}.{ext}"
        fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
        paths.append(target)
    plt.close(fig)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Validation-only redraw of the supplied sketch; existing experiment artifacts unchanged",
              "new_model_calls": 0, "candidate_count": sum(len(r["candidates"]) for r in runs),
              "branch_data": runs, "pooled_maxima": maxima,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
              "renderer_sha256": sha(Path(__file__)),
              "outputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              "text_within_canvas": True, "manual_visual_review": "pending"}
    (OUT / "validation_flow_provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in paths], "candidate_count": 16, "new_model_calls": 0, "pooled_maxima": maxima}, ensure_ascii=False))


if __name__ == "__main__":
    render()
