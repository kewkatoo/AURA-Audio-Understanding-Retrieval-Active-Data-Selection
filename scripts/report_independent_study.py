"""Paired, recording-aligned comparisons and plots for the frozen v5 runs."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aura.cache.store import file_content_hash
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap


def paired_comparisons(predictions, references):
    frame = predictions.copy()
    frame["correct"] = frame.prediction.astype(str).eq(frame.label.astype(str)).astype(float)
    if frame.duplicated(["strategy", "budget", "seed", "id"]).any():
        raise ValueError("Duplicate prediction keys")
    rows = []
    for budget in sorted(frame.budget.unique()):
        at_budget = frame[frame.budget == budget]
        for reference in references:
            base = at_budget[at_budget.strategy == reference]
            if base.empty:
                continue
            for strategy, group in at_budget.groupby("strategy"):
                if strategy == reference:
                    continue
                keys = ["seed", "id"]
                aligned = group.merge(base, on=keys, suffixes=("", "_base"), validate="one_to_one")
                if len(aligned) != len(group) or len(aligned) != len(base):
                    raise ValueError("Paired cohorts or seeds differ")
                if not (aligned.label.astype(str).eq(aligned.label_base.astype(str)).all()
                        and aligned.speaker_id.eq(aligned.speaker_id_base).all()):
                    raise ValueError("Paired labels/speakers differ")
                aligned["delta"] = aligned.correct - aligned.correct_base
                per_record = aligned.groupby("id").agg(delta=("delta", "mean"), speaker=("speaker_id", "first"))
                low, high = speaker_bootstrap(per_record.delta.to_numpy(), per_record.speaker.to_numpy())
                rows.append(dict(strategy=strategy, reference=reference, budget=int(budget),
                                 difference_pp=100 * per_record.delta.mean(), low_pp=low, high_pp=high,
                                 speakers=per_record.speaker.nunique()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    root = Path("results/duplicate_control_v5" if args.control else "results/audiomnist_v5")
    predictions = pd.read_csv(root / "predictions.csv.gz", dtype={"label": str, "prediction": str})
    runs = pd.read_csv(root / "runs.csv")
    if args.control:
        prior = pd.read_csv(root / "reference_predictions.csv.gz")
        prior = prior[prior.strategy.isin(["random", "representative", "quality_representative"])]
        predictions = pd.concat([predictions, prior], ignore_index=True)
        old = pd.read_csv("results/scaling_v4/runs.csv")
        old = old.rename(columns={"holdout_accuracy": "accuracy", "holdout_macro_f1": "macro_f1"})
        runs = pd.concat([runs, old[old.strategy.isin(prior.strategy.unique())]], ignore_index=True)
    comparisons = paired_comparisons(predictions, ["random", "deduplicated_random", "representative"])
    comparisons.to_csv(root / "speaker_bootstrap.csv", index=False)
    summary = runs.groupby(["strategy", "budget"]).accuracy.agg(["mean", "std"])
    summary.to_csv(root / "comparison_summary.csv")
    runs.groupby(["strategy", "budget"])[["selection_seconds", "fit_seconds"]].mean().to_csv(root / "observed_costs.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in summary.reset_index().groupby("strategy"):
        if strategy == "full_pool":
            ax.axhline(group["mean"].iloc[0] * 100, color="black", linestyle="--", label="Full pool")
        else:
            ax.plot(group.budget, group["mean"] * 100, marker="o", label=strategy.replace("_", " "))
    ax.set(xscale="log", xlabel="Selected recordings (log scale)", ylabel="Held-out accuracy (%)",
           title="Speech Commands: duplicate control" if args.control else "AudioMNIST: frozen selector transfer")
    ax.set_xticks([100, 200, 400, 800, 1600], labels=[100, 200, 400, 800, 1600])
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.svg")
    primary = comparisons[(comparisons.strategy == "representative") &
                          (comparisons.reference == "deduplicated_random")]
    lines = ["# " + ("Duplicate-control results" if args.control else "Independent AudioMNIST results"), "",
             "Mean held-out accuracy across five fixed selection seeds:", "",
             "| Method | Budget | Accuracy |", "|---|---:|---:|"]
    for row in summary.reset_index().itertuples():
        lines.append(f"| {row.strategy} | {row.budget} | {row.mean * 100:.2f}% |")
    lines += ["", "Representative minus deduplicated random (percentage points; paired speaker bootstrap):", "",
              "| Budget | Difference | Exploratory 95% interval |", "|---:|---:|---:|"]
    for row in primary.itertuples():
        lines.append(f"| {row.budget} | {row.difference_pp:+.2f} | [{row.low_pp:+.2f}, {row.high_pp:+.2f}] |")
    if args.control:
        lines += ["", "This is a post-hoc diagnostic on the already observed v4 holdout. It is not fresh confirmation.",
                  "Deduplication samples uniformly among unique raw-byte contents, retaining the smallest ID per content.",
                  "This also changes content sampling weights and the seed-to-subset mapping; it is not a pure paired removal of duplicates.",
                  "Exact-byte matching does not identify all acoustically similar recordings."]
    else:
        lines += ["", "Dataset: [AudioMNIST](https://github.com/soerenab/AudioMNIST), revision 630d7dab4c040882834de6fa21baf9a60372accd (MIT).",
                  "Citation: Becker et al., AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark, DOI 10.1016/j.jfranklin.2023.11.038.",
                  "A declared 45/15 speaker split uses 22,500 training and 7,500 held-out recordings before any content exclusions. See partition.json for exact accepted counts.",
                  "This custom split is not directly comparable with the authors' published benchmark. Digits and a different recording environment test transfer, but remain speech classification.",
                  "All classifier and selector settings transfer from Speech Commands; no new dataset tuning is performed. Full-pool C=10 and subset C values differ by the previously locked choices.",
                  "Class labels are already known during selection. This measures subset training accuracy, not label-acquisition savings."]
    lines += ["", "Intervals resample held-out speakers and condition on the five fixed models. They do not include hyperparameter/search uncertainty, have no multiplicity adjustment, and should be treated as exploratory.",
              "Comparisons evaluate frozen selector-plus-classifier recipes: previously tuned C can differ by method/budget, so all differences cannot be attributed solely to selection. At budget 100, random, deduplicated random, and representatives share C=1.",
              "Selection uses frozen wav2vec2 embeddings for every candidate. Head-training reductions alone do not establish total compute savings."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "analysis_manifest.json").write_text(json.dumps(dict(script_hash=file_content_hash(__file__),
        protocol_hash=file_content_hash(root / "protocol.json"), runs_hash=file_content_hash(root / "runs.csv"),
        predictions_hash=file_content_hash(root / "predictions.csv.gz")), indent=2))
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
