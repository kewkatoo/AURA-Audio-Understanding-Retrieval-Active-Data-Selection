#!/usr/bin/env python
"""Validation-tuned linear probes for offline label-aware subset selection.

All configurations are locked before test data are loaded. Reports include
selection IDs, per-class metrics, train/selection costs, and source hashes.
The full baseline means the entire configured candidate pool, not necessarily
the entire available Speech Commands training split.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aura.cache.store import file_content_hash
from aura.config import AuraConfig
from aura.ingest.metadata import load_embedding_matrix
from aura.provenance import PIPELINE_VERSION, source_fingerprint, runtime_versions
from aura.selection.combined import CombinedSelector
from aura.selection.diversity import DiversitySelector
from aura.selection.quality import QualitySelector
from aura.selection.random import RandomSelector
from aura.selection.utils import build_embedding_lookup


def load_split(config: AuraConfig, split: str):
    manifest_path = Path(config.paths.metadata_dir) / f"{split}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["pipeline_version"] != PIPELINE_VERSION or manifest["source_fingerprint"] != source_fingerprint():
        raise ValueError(f"Regenerate {split}: artifacts were made by a different pipeline")
    for path, expected in manifest["artifacts"].items():
        if file_content_hash(path) != expected:
            raise ValueError(f"Artifact does not match manifest: {path}")
    metadata = pd.read_parquet(Path(config.paths.metadata_dir) / f"{split}.parquet")
    matrix, index = load_embedding_matrix(Path(config.paths.embeddings_dir) / f"{split}.npy",
                                          Path(config.paths.embeddings_dir) / f"{split}_index.parquet")
    lookup = build_embedding_lookup(metadata, matrix, index)
    aligned = np.stack([lookup[sid] for sid in metadata.id])
    if not metadata.split.eq(split).all() or not np.isfinite(aligned).all():
        raise ValueError(f"Invalid {split} artifacts")
    return metadata, aligned, manifest


def assert_disjoint(a: pd.DataFrame, b: pd.DataFrame):
    if set(a.id) & set(b.id) or set(a.speaker_id.dropna()) & set(b.speaker_id.dropna()):
        raise ValueError("Sample or speaker overlap between experiment splits")


def fit_probe(x, y, c):
    # Fit scaling ONLY on selected training examples, never validation/test.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=2000))
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model, time.perf_counter() - started


def metrics(y, predictions, labels):
    return {"accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.v2.yaml")
    parser.add_argument("--output-dir", default="results/benchmark_v2")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1, 10])
    args = parser.parse_args()
    if (any(b < 1 for b in args.budgets) or any(c <= 0 for c in args.c_values)
            or any(not 0 <= a <= 1 for a in args.alphas)
            or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("Invalid budgets, C values, alphas, or duplicate seeds")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # An existing completed experiment is immutable unless a new directory is used.
    if (out / "experiment.json").exists():
        raise ValueError("Choose a new output directory; this experiment is already complete")
    config = AuraConfig.from_yaml(args.config)
    started = time.perf_counter()
    train, x_train, train_manifest = load_split(config, "train")
    val, x_val, val_manifest = load_split(config, "validation")
    assert_disjoint(train, val)
    if any(b > len(train) for b in args.budgets):
        raise ValueError("Experiment budgets may not exceed the candidate pool")
    if train_manifest["encoder"] != val_manifest["encoder"]:
        raise ValueError("Splits must use the same encoder")
    labels = sorted(train.label.unique())
    if set(val.label) != set(labels):
        raise ValueError("Validation labels differ from training labels")
    y_train, y_val = train.label.to_numpy(), val.label.to_numpy()
    embeddings = dict(zip(train.id, x_train))
    row_for_id = dict(zip(train.id, range(len(train))))
    sweep = []
    finalists = []
    selections = out / "selections"
    selections.mkdir(exist_ok=True)
    jobs = [(s, b) for b in sorted(set(args.budgets))
            for s in ["random", "quality", "diversity", "combined"]] + [("full_pool", len(train))]
    for strategy, budget in jobs:
        candidates = []
        alphas = sorted(set(args.alphas)) if strategy == "combined" else [-1.0]
        # The full-pool baseline and quality-only method are deterministic;
        # repeats across selection seeds do not imply independent head runs.
        seeds = args.seeds if strategy != "full_pool" else [args.seeds[0]]
        for alpha in alphas:
            for seed in seeds:
                selection_started = time.perf_counter()
                if strategy == "full_pool":
                    ids = train.id.tolist()
                    selection_record = {"strategy": strategy, "selected_ids": ids, "seed": seed}
                else:
                    selector = {"random": RandomSelector, "quality": QualitySelector,
                                "diversity": DiversitySelector, "combined": lambda: CombinedSelector(alpha)}[strategy]()
                    result = selector.select(train, embeddings, budget, seed)
                    ids, selection_record = result.selected_ids, result.to_dict()
                selection_seconds = time.perf_counter() - selection_started
                rows = [row_for_id[sid] for sid in ids]
                selection_path = selections / f"{strategy}_budget{budget}_seed{seed}_alpha{alpha:g}.json"
                selection_path.write_text(json.dumps(selection_record, indent=2))
                for c in sorted(set(args.c_values)):
                    model, fit_seconds = fit_probe(x_train[rows], y_train[rows], c)
                    scores = metrics(y_val, model.predict(x_val), labels)
                    record = {"strategy": strategy, "budget": budget, "seed": seed,
                              "alpha": alpha, "C": c, "validation_accuracy": scores["accuracy"],
                              "validation_macro_f1": scores["macro_f1"],
                              "selection_seconds": selection_seconds, "fit_seconds": fit_seconds,
                              "mean_quality": float(train.iloc[rows].quality_score.mean()),
                              "distinct_speakers": int(train.iloc[rows].speaker_id.nunique()),
                              "selection_path": str(selection_path)}
                    sweep.append(record)
                    candidates.append((record, model))
        scores = pd.DataFrame([r for r, _ in candidates])
        rank = scores.groupby(["alpha", "C"], as_index=False)[["validation_accuracy", "validation_macro_f1"]].mean()
        rank = rank.sort_values(["validation_accuracy", "validation_macro_f1", "alpha", "C"],
                                ascending=[False, False, True, True])
        best = rank.iloc[0]
        finalists.extend((r, model) for r, model in candidates if r["alpha"] == best.alpha and r["C"] == best.C)
        pd.DataFrame(sweep).to_csv(out / "validation_sweep.csv", index=False)
        print(f"Locked {strategy} B={budget}: alpha={best.alpha:g}, C={best.C:g}, val={best.validation_accuracy:.4f}", flush=True)

    # Freeze every choice BEFORE opening test artifacts.
    choices = [{k: r[k] for k in ["strategy", "budget", "seed", "alpha", "C", "selection_path"]}
               for r, _ in finalists]
    (out / "validation_choices.json").write_text(json.dumps(choices, indent=2))
    test, x_test, test_manifest = load_split(config, "test")
    assert_disjoint(train, test)
    assert_disjoint(val, test)
    if train_manifest["encoder"] != test_manifest["encoder"] or set(test.label) != set(labels):
        raise ValueError("Test encoder/labels differ from training")
    runs = []
    per_class = []
    y_test = test.label.to_numpy()
    for record, model in finalists:
        predictions = model.predict(x_test)
        scores = metrics(y_test, predictions, labels)
        run = {**record, "test_accuracy": scores["accuracy"], "test_macro_f1": scores["macro_f1"]}
        runs.append(run)
        recalls = recall_score(y_test, predictions, labels=labels, average=None, zero_division=0)
        per_class.append({"strategy": record["strategy"], "budget": record["budget"], "seed": record["seed"],
                          "recall": dict(zip(labels, recalls.tolist())),
                          "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()})
    df = pd.DataFrame(runs)
    df.to_csv(out / "runs.csv", index=False)
    summary = df.groupby(["strategy", "budget"], as_index=False).agg(
        seeds=("seed", "count"), accuracy_mean=("test_accuracy", "mean"),
        accuracy_std=("test_accuracy", "std"), macro_f1_mean=("test_macro_f1", "mean"),
        macro_f1_std=("test_macro_f1", "std"), selection_seconds_mean=("selection_seconds", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), alpha=("alpha", "first"), C=("C", "first"))
    summary.to_csv(out / "summary.csv", index=False)
    (out / "per_class.json").write_text(json.dumps({"labels": labels, "runs": per_class}, indent=2))
    manifest = {"config": asdict(config), "arguments": vars(args), "pipeline_version": PIPELINE_VERSION,
                "source_fingerprint": source_fingerprint(), "runtime_versions": runtime_versions(),
                "sklearn_version": sklearn.__version__, "script_hash": file_content_hash(__file__),
                "split_manifests": {"train": train_manifest, "validation": val_manifest, "test": test_manifest},
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "Offline label-aware selection on the configured candidate pool; frozen Wav2Vec2 linear probe"}
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Corrected-pipeline selection benchmark", "",
             f"Candidate pool: {len(train)}; validation: {len(val)}; test: {len(test)}.", "",
             "StandardScaler + multinomial logistic regression, fitted only on each training subset. "
             "C and combined alpha were selected by mean validation accuracy across selection seeds; "
             "ties use macro-F1, then smaller alpha/C. Test data were loaded after choices were saved.", "",
             "| Strategy | Budget | Test accuracy ± seed std | Macro-F1 | Alpha | C |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        std = f"{100 * row.accuracy_std:.2f}" if np.isfinite(row.accuracy_std) else "n/a"
        lines.append(f"| {row.strategy} | {row.budget} | {100 * row.accuracy_mean:.2f}% ± {std} pp | "
                     f"{row.macro_f1_mean:.4f} | {row.alpha if row.alpha >= 0 else '—'} | {row.C:g} |")
    lines += ["", "Standard deviations describe selection-seed variation, not independent test datasets. "
              "Quality-only and the classifier optimizer are deterministic. The full-pool baseline is "
              "the configured candidate pool, not all available training recordings.", "",
              "Preparation timings are stored per split in the input manifests; per-run selection/head fit "
              "timings are in runs.csv. These measurements do not establish a total compute saving over "
              "training a model directly on raw audio."]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
