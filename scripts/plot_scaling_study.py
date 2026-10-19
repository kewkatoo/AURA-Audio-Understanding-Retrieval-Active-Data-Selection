"""Plot a completed scaling study and its paired speaker-bootstrap differences."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/scaling_v4")
    args = parser.parse_args()
    root = Path(args.results)
    summary = pd.read_csv(root / "summary.csv")
    differences = pd.read_csv(root / "speaker_bootstrap.csv")
    experiment = json.loads((root / "experiment.json").read_text())
    styles = {"random": ("Random", "#333333", "--"), "quality": ("Quality", "#999999", ":"),
              "diversity": ("Farthest point", "#d88119", "-"), "combined": ("Original combined", "#9866b6", "-"),
              "representative": ("Cluster representatives", "#2563b4", "-"),
              "quality_representative": ("Quality gate + representatives", "#00877c", "-")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    budgets = sorted(summary[summary.strategy != "full_pool"].budget.unique())
    for strategy, (label, color, style) in styles.items():
        rows = summary[summary.strategy == strategy].sort_values("budget")
        if rows.empty:
            continue
        axes[0].errorbar(rows.budget, rows.accuracy * 100, yerr=rows["std"].fillna(0)*100,
                         label=label, color=color, linestyle=style, marker="o", markersize=4, capsize=3)
    full = summary[summary.strategy == "full_pool"].iloc[0]
    axes[0].axhline(full.accuracy*100, color="#666666", linestyle="-.", linewidth=1,
                   label=f"Full {int(full.budget):,}-example pool")
    axes[0].set_ylabel("Fresh-holdout accuracy (%)")
    axes[0].set_title("Fixed methods, larger candidate pool", fontsize=11)
    for strategy in ["combined", "representative", "quality_representative"]:
        rows = differences[differences.strategy == strategy].sort_values("budget")
        label, color, _ = styles[strategy]
        axes[1].plot(rows.budget, rows.delta_pp, color=color, marker="o", markersize=4, label=label)
        axes[1].fill_between(rows.budget.to_numpy(), rows.bootstrap_low_pp.to_numpy(),
                              rows.bootstrap_high_pp.to_numpy(), color=color, alpha=0.1)
    axes[1].axhline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].set_ylabel("Accuracy difference from random (percentage points)")
    axes[1].set_title("Paired 95% speaker-bootstrap intervals", fontsize=11)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(x) for x in budgets])
        ax.set_xlabel("Selected training examples")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Scaling audio selection: {experiment['all_candidate_count']:,} candidates, "
                 f"{experiment['holdout_count']:,} fresh holdout recordings", fontsize=14, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.015, "Frozen v3 settings · five selection seeds · left bars: seed std · right intervals: exploratory, conditional, unadjusted",
             ha="center", color="#555555", fontsize=8)
    fig.tight_layout(rect=(0, 0.23, 1, 0.93))
    for extension in ["png", "svg"]:
        fig.savefig(root / f"scaling_curves.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
