"""Refit locked heads and quantify paired differences by resampling speakers.

This is post-confirmation analysis only: it cannot change selected IDs, C,
alpha, or quality thresholds. Intervals condition on the fitted selections
and do not include uncertainty from tuning, dataset choice, or future domains.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from aura.cache.store import file_content_hash
from aura.config import AuraConfig
from aura.ingest.metadata import load_embedding_matrix
from aura.selection.utils import build_embedding_lookup
from scripts.evaluate_selection import load_split, fit_probe


def speaker_bootstrap(delta, speakers, replicates=2000, seed=0):
    """Paired accuracy difference: pool recordings within each resampled speaker."""
    if replicates < 1 or len(delta) != len(speakers) or not np.isfinite(delta).all():
        raise ValueError("Invalid bootstrap inputs")
    table = pd.DataFrame({"delta": delta, "speaker": speakers}).groupby("speaker").delta.agg(["sum", "count"])
    if len(table) < 2:
        raise ValueError("At least two confirmation speakers are required")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(table), size=(replicates, len(table)))
    estimates = table["sum"].to_numpy()[sample].sum(axis=1) / table["count"].to_numpy()[sample].sum(axis=1)
    return tuple(float(x * 100) for x in np.percentile(estimates, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/representative_v3")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    directory = Path(args.results)
    if (directory / "uncertainty.json").exists():
        raise ValueError("Uncertainty analysis is already complete")
    study = json.loads((directory / "experiment.json").read_text())
    config = AuraConfig.from_yaml(study["arguments"]["config"])
    train, x_train, _ = load_split(config, "train")
    confirmation_dir = directory / "confirmation"
    manifest = json.loads((confirmation_dir / "manifest.json").read_text())
    for path, expected in manifest["artifacts"].items():
        if file_content_hash(path) != expected:
            raise ValueError(f"Changed confirmation artifact: {path}")
    confirmation = pd.read_parquet(confirmation_dir / "metadata.parquet")
    matrix, index = load_embedding_matrix(confirmation_dir / "confirmation.npy", confirmation_dir / "index.parquet")
    lookup = build_embedding_lookup(confirmation, matrix, index)
    x_confirmation = np.stack([lookup[sid] for sid in confirmation.id])
    y_confirmation = confirmation.label.to_numpy()
    train_rows = dict(zip(train.id, range(len(train))))
    runs = pd.read_csv(directory / "runs.csv")
    predictions = []
    correct = {}
    models = {}
    selection_hashes = {}
    for row in runs.itertuples():
        selection_path = Path(row.selection_path)
        selection_hashes[str(selection_path)] = file_content_hash(selection_path)
        ids = json.loads(selection_path.read_text())["selected_ids"]
        key = (tuple(ids), row.C)
        if key not in models:
            rows = [train_rows[sid] for sid in ids]
            models[key], _ = fit_probe(x_train[rows], train.label.to_numpy()[rows], row.C)
        guessed = models[key].predict(x_confirmation)
        values = (guessed == y_confirmation).astype(np.float64)
        if not np.isclose(values.mean(), row.confirmation_accuracy, atol=1e-10):
            raise ValueError("Locked-model refit does not reproduce the original confirmation accuracy")
        correct[(row.strategy, row.budget, row.seed)] = values
        predictions.extend({"strategy": row.strategy, "budget": row.budget, "seed": row.seed,
                            "id": sid, "speaker_id": speaker, "label": label, "prediction": pred,
                            "correct": int(value)} for sid, speaker, label, pred, value in
                           zip(confirmation.id, confirmation.speaker_id, y_confirmation, guessed, values))
    pd.DataFrame(predictions).to_csv(directory / "confirmation_predictions.csv", index=False)
    differences = []
    for (strategy, budget), group in runs[runs.strategy != "full_pool"].groupby(["strategy", "budget"]):
        candidate = np.mean([correct[(strategy, budget, seed)] for seed in group.seed], axis=0)
        random = np.mean([correct[("random", budget, seed)] for seed in group.seed], axis=0)
        delta = candidate - random
        low, high = speaker_bootstrap(delta, confirmation.speaker_id, args.replicates, args.seed)
        differences.append({"strategy": strategy, "budget": int(budget), "difference_pp": float(delta.mean()*100),
                            "speaker_bootstrap_low_pp": low, "speaker_bootstrap_high_pp": high,
                            "speakers": int(confirmation.speaker_id.nunique())})
    frame = pd.DataFrame(differences)
    frame.to_csv(directory / "speaker_bootstrap.csv", index=False)
    (directory / "uncertainty.json").write_text(json.dumps({
        "arguments": vars(args), "script_hash": file_content_hash(__file__),
        "study_manifest_hash": file_content_hash(directory / "experiment.json"),
        "selection_hashes": selection_hashes, "all_original_accuracies_reproduced": True,
        "method": "paired speaker-cluster percentile bootstrap of recording accuracy, averaged over locked seed models",
        "limitation": "conditional on selections/tuning; no multiple-comparison correction; same small cohort",
    }, indent=2))
    lines = ["# Confirmation uncertainty", "",
             "Each locked model was refitted and its original confirmation accuracy reproduced before analysis. "
             "No setting or selected ID was changed. Intervals resample speakers with replacement and count "
             "all their recordings together; seed models are averaged within each recording.", "",
             "| Strategy | Budget | Difference from random | 95% speaker-bootstrap interval |",
             "|---|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {row.strategy} | {row.budget} | {row.difference_pp:+.2f} pp | "
                     f"[{row.speaker_bootstrap_low_pp:+.2f}, {row.speaker_bootstrap_high_pp:+.2f}] pp |")
    lines += ["", "These exploratory intervals condition on the existing training/tuning experiment. "
              "They do not account for hyperparameter search, multiple comparisons, or a new dataset/domain. "
              "They are not a reason to tune further against this confirmation cohort."]
    (directory / "UNCERTAINTY.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
