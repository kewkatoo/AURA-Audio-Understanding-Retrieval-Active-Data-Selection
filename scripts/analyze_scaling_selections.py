"""Post-study redundancy and efficiency diagnostics; does not fit new models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/scaling_v4")
    parser.add_argument("--data-root", default="data/scaling_v4")
    args = parser.parse_args()
    output, root = Path(args.results), Path(args.data_root)
    if not (output / "experiment.json").exists():
        raise ValueError("Complete the frozen benchmark before post-study interpretation")
    plan = json.loads((root / "partition.json").read_text())
    hashes = {row["sample_id"]: row["audio_hash"] for row in plan["train"]}
    runs = pd.read_csv(output / "runs.csv")
    rows = []
    for run in runs.itertuples():
        ids = json.loads(Path(run.selection_path).read_text())["selected_ids"]
        rows.append({"strategy": run.strategy, "budget": run.budget, "seed": run.seed,
                     "selected_records": len(ids), "unique_audio_contents": len({hashes[sid] for sid in ids}),
                     "duplicate_contents_selected": len(ids)-len({hashes[sid] for sid in ids})})
    redundancy = pd.DataFrame(rows)
    redundancy.to_csv(output / "selection_redundancy.csv", index=False)
    summary = redundancy.groupby(["strategy", "budget"], as_index=False).agg(
        duplicate_contents_mean=("duplicate_contents_selected", "mean"),
        duplicate_contents_max=("duplicate_contents_selected", "max"))
    summary.to_csv(output / "redundancy_summary.csv", index=False)
    results = pd.read_csv(output / "summary.csv")
    full = results[results.strategy == "full_pool"].iloc[0]
    efficiency = results[results.strategy != "full_pool"].copy()
    efficiency["candidate_fraction"] = efficiency.budget / full.budget
    efficiency["gap_to_full_pool_pp"] = 100 * (full.accuracy - efficiency.accuracy)
    efficiency["measured_select_plus_fit_seconds"] = efficiency.selection_seconds + efficiency.fit_seconds
    efficiency = efficiency.merge(summary, on=["strategy", "budget"])
    efficiency.to_csv(output / "efficiency_summary.csv", index=False)
    lines = ["# Redundancy and data-efficiency diagnostics", "",
             "These post-study diagnostics do not change model choices or selections. "
             "Candidate fractions describe recording-count reduction, not annotation savings.", "",
             "| Strategy | Budget | Candidate fraction | Gap to full-pool accuracy | Duplicate contents selected, mean |",
             "|---|---:|---:|---:|---:|"]
    for row in efficiency.itertuples():
        lines.append(f"| {row.strategy} | {row.budget} | {100*row.candidate_fraction:.2f}% | "
                     f"{row.gap_to_full_pool_pp:.2f} pp | {row.duplicate_contents_mean:.1f} |")
    lines += ["", "Raw-content duplicates can partly explain differences between selectors. A future "
              "deduplicated-random control is needed to isolate representation coverage from simple deduplication. "
              "Do not tune that control against this now-observed holdout.", "",
              "Selection and fit times include a mixture of sequential and four-worker execution. "
              "They are observed wall times with contention, not a controlled algorithm-speed comparison. "
              "Preparation costs must also be counted before claiming compute savings."]
    (output / "EFFICIENCY.md").write_text("\n".join(lines), encoding="utf-8")
    print(efficiency[["strategy", "budget", "candidate_fraction", "gap_to_full_pool_pp", "duplicate_contents_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
