"""Standalone publication-exportable plots using an existing matplotlib Python."""
import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
with (root / "checkpoint_summary.csv").open(encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
noise = json.loads((root / "noise_reference.json").read_text(encoding="utf-8"))
colors = {"0": "#2878b5", "100": "#d95f02", "50": "#248558"}
for kind in ["cost_performance", "round_gain"]:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for run in dict.fromkeys(row["run_id"] for row in rows):
        group = [row for row in rows if row["run_id"] == run]
        x = [float(r["C_usd"] if kind == "cost_performance" else r["T"]) for r in group]
        y = [float(r["S_pp"]) for r in group]
        color = colors[group[0]["D"]]
        if run.endswith("b"):
            color = {"0": "#79a9d1", "100": "#ed9959"}.get(group[0]["D"], color)
        ax.plot(x, y, marker="s" if run.endswith("b") else "o", color=color, linestyle="--" if run.endswith("b") else "-", label=run)
    ax.axhline(noise["S0_pp"], color="#777777", linestyle=":", label="Frozen R4B reference")
    ax.set(xlabel="Cumulative algorithm cost (logged USD)" if kind == "cost_performance" else "Additional evolution rounds T (N = 2T)", ylabel="Test accuracy of score-selected harness (%)")
    if kind == "round_gain":
        ax.set_xticks([1, 2])
        d0_a = [float(r["S_pp"]) for r in rows if r["run_id"] == "D0_a"]
        d0_b = [float(r["S_pp"]) for r in rows if r["run_id"] == "D0_b"]
        if d0_a == d0_b and len(set(d0_a)) == 1:
            ax.annotate(f"D0_a = D0_b: {d0_a[0]:g}%", (1.5, d0_a[0]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=colors["0"])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(root / f"{kind}.{ext}", dpi=200)
    plt.close(fig)
