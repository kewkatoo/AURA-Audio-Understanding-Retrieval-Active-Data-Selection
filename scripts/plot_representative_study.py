"""Render development/confirmation curves from a completed representative study."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/representative_v3")
    args = parser.parse_args()
    directory = Path(args.results)
    data = pd.read_csv(directory / "summary.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    styles = {"random": ("Random", "#333333", "--"),
              "quality": ("Quality", "#969696", ":"),
              "diversity": ("Farthest point", "#d88119", "-"),
              "combined": ("Original combined", "#9866b6", "-"),
              "representative": ("Cluster representatives", "#2563b4", "-"),
              "quality_representative": ("Quality gate + representatives", "#00877c", "-")}
    for ax, column, title in zip(axes, ["validation_accuracy", "confirmation_accuracy"],
                                 ["Development: settings chosen here", "Fresh speakers: settings already locked"]):
        for strategy, (label, color, linestyle) in styles.items():
            rows = data[data.strategy == strategy].sort_values("budget")
            error = rows.confirmation_std.fillna(0) * 100 if column == "confirmation_accuracy" else None
            ax.errorbar(rows.budget, rows[column] * 100, yerr=error, label=label,
                        color=color, linestyle=linestyle, marker="o", markersize=4, capsize=3)
        reference = data[data.strategy == "full_pool"].iloc[0]
        ax.axhline(reference[column] * 100, color="#666666", linewidth=1, linestyle="-.",
                   label=f"Full {int(reference.budget):,}-example pool")
        ax.set_xscale("log", base=2)
        budgets = sorted(data[data.strategy != "full_pool"].budget.unique())
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_xlabel("Selected training examples")
        ax.set_title(title, fontsize=11, pad=12)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Classifier accuracy (%)")
    fig.suptitle("Does representative coverage improve audio subset selection?", fontsize=15, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.015, "Frozen Wav2Vec2 + linear probe · five selection seeds · error bars: seed standard deviation, not confidence intervals",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.21, 1, 0.93))
    for extension in ["png", "svg"]:
        fig.savefig(directory / f"accuracy_curves.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
