"""Generate the figures from the saved test results and trajectory logs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "artifacts" / "debt_research"
FIGURES = HERE / "figures"
COLORS = {"work": "#006F74", "timeout": "#A24C00"}


def outcomes(results):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.45), layout="constrained")
    groups = (("urgency_0.25", "Work, 0.25", -0.22, "#7DB2AE"),
              ("urgency_1", "Work, 1.0", 0, COLORS["work"]),
              ("mixed_demos_urgency_1", "Work and casino, 1.0", 0.22, COLORS["timeout"]))
    seeds = (11, 29, 47)
    x = np.arange(len(seeds))
    for ax, metric, label in zip(axes, ("success", "debt_cleared"), ("Taxi bought out", "Debt cleared")):
        for prefix, name, offset, color in groups:
            counts = [round(results[f"{prefix}_seed_{seed}"]["summary"][metric] * 128) for seed in seeds]
            ax.bar(x + offset, counts, width=.20, label=name, color=color)
        ax.set(title=label, xticks=x, xticklabels=seeds, ylim=(0, 128), yticks=(0, 32, 64, 96, 128),
               xlabel="Training seed")
        ax.grid(axis="y", alpha=.24)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Episodes out of 128")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(FIGURES / "debt-outcomes.png", dpi=165)
    plt.close(fig)


def trajectories():
    cases = (
        ("mixed_demos_urgency_1_seed_11_first_win_91010.npz", "Buyout, training seed 11", COLORS["work"]),
        ("mixed_demos_urgency_1_seed_29_work_sleep_and_repeat_91010.npz", "Timeout, training seed 29", COLORS["timeout"]),
    )
    fig, (debt, cash) = plt.subplots(2, 1, figsize=(9.2, 4.4), sharex=True, layout="constrained")
    for file, label, color in cases:
        with np.load(DATA / "cases" / file, allow_pickle=False) as row:
            for ax, metric in ((debt, "debt"), (cash, "cash")):
                ax.step(row["time"], row[metric], where="post", color=color, lw=1.6, label=label)
    for ax in (debt, cash):
        ax.axvspan(99, 132, color="#DAA450", alpha=.24)
        ax.grid(alpha=.20)
        ax.set_axisbelow(True)
    debt.set(ylabel="Debt, USD", title="Same two wagers on test seed 91010")
    debt.text(147, 790, "−$55 at tick 99\n+$270 at tick 132", va="top", fontsize=9)
    cash.set(ylabel="Cash, USD", xlabel="Campaign tick")
    cash.axhline(115, color="#52565A", ls="--", lw=.8)
    cash.text(1535, 123, "Buyout price", ha="right", va="bottom", fontsize=8)
    debt.legend(loc="upper right", frameon=False)
    fig.savefig(FIGURES / "same-bets-different-outcomes.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    outcomes(json.loads((DATA / "results.json").read_text(encoding="utf-8")))
    trajectories()
