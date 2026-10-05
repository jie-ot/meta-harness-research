"""Render additional standalone figures from saved CSVs; no model calls."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"


def read_rows(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def frozen_repeats():
    rows = read_rows("noise_summary.csv")
    labels = ["R4B", "R7A", "R12B"]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.1), sharey=True, constrained_layout=True)
    for ax, field, title in zip(axes, ["score_pp", "audit_pp"], ["Fixed score panel", "External test panel"]):
        for rep, marker, color, offset in [(1, "o", "#2878b5", -.09), (2, "s", "#d95f02", .09)]:
            values = [float(next(r[field] for r in rows if r["version"] == label and int(r["repetition"]) == rep)) for label in labels]
            x = [i + offset for i in range(len(labels))]
            ax.scatter(x, values, color=color, marker=marker, s=48, label=f"Repeat {rep}", zorder=3)
            for xx, value in zip(x, values):
                ax.annotate(f"{value:g}", (xx, value), xytext=(0, 7 if rep == 1 else -13), textcoords="offset points", ha="center", fontsize=9, color=color)
        ax.set(title=title, xticks=range(3), xticklabels=labels, ylim=(0, 55), xlim=(-.45, 2.45), xlabel="Frozen harness version")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Exact-match accuracy (%)")
    fig.suptitle("Frozen re-evaluation: 100 items per panel, local response cache off", fontsize=11)
    save(fig, "frozen_repeats")


def comparable_cost():
    path = OUT / "checkpoint_cost_detail.csv"
    if not path.exists():
        return
    rows = read_rows(path.name)
    colors = {"0":"#2878b5", "100":"#d95f02"}
    fig, ax = plt.subplots(figsize=(8.2, 5), constrained_layout=True)
    for run in dict.fromkeys(r["run_id"] for r in rows):
        group = [r for r in rows if r["run_id"] == run]
        x = [float(r["experimental_C_usd"]) for r in group]
        y = [float(r["S_pp"]) for r in group]
        color = colors[group[0]["D"]] if not run.endswith("b") else {"0":"#79a9d1", "100":"#ed9959"}[group[0]["D"]]
        ax.plot(x, y, marker="s" if run.endswith("b") else "o", color=color, linestyle="--" if run.endswith("b") else "-", label=run)
        for xx, yy, r in zip(x, y, group):
            ax.annotate(f"T={r['T']}", (xx, yy), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Experimental cost C (logged USD; setup failures listed separately)", ylabel="Test accuracy of score-selected harness (%)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    save(fig, "comparable_cost_performance")


if __name__ == "__main__":
    frozen_repeats()
    comparable_cost()
