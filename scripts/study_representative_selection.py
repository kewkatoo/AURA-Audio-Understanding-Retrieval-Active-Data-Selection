#!/usr/bin/env python
"""Compare representative selectors using tuning validation and fresh speakers.

The previously inspected test set is never opened. All settings are locked on
the existing validation set before unused validation speakers are processed.
Experimental algorithms live outside the frozen v2 preparation package so
adding a selector does not invalidate otherwise unchanged audio artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, recall_score

from aura.cache.store import CacheStore, file_content_hash
from aura.config import AuraConfig, PathsConfig
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.ingest.dataset import FilteredSpeechCommands
from aura.ingest.metadata import save_embedding_matrix, save_metadata_parquet
from aura.provenance import source_fingerprint, runtime_versions
from aura.selection.combined import CombinedSelector
from aura.selection.diversity import DiversitySelector
from aura.selection.quality import QualitySelector
from aura.selection.random import RandomSelector
from experiments.representative import RepresentativeSelector, QualityRepresentativeSelector, coverage_diagnostics
from scripts.evaluate_selection import load_split, assert_disjoint, fit_probe, metrics
from scripts.prepare_data import load_training_stats, process_split


def prepare_confirmation(config, train, tuning, output, expected_encoder):
    if not (output / "locked_choices.json").exists():
        raise ValueError("Lock choices before preparing confirmation data")
    excluded = set(train.speaker_id.dropna()) | set(tuning.speaker_id.dropna())
    records = [r for r in FilteredSpeechCommands(config.dataset.root, config.dataset.classes,
                                                "validation", download=False).records()
               if r.speaker_id not in excluded]
    records.sort(key=lambda r: r.sample_id)
    if not records:
        raise ValueError("No unused official-validation speakers remain")
    cohort = [{"id": r.sample_id, "speaker_id": r.speaker_id, "label": r.label,
               "audio_hash": file_content_hash(r.audio_path)} for r in records]
    (output / "confirmation_cohort.json").write_text(json.dumps(cohort, indent=2))
    norm_stats = load_training_stats(config)
    config = AuraConfig(dataset=config.dataset, audio=config.audio, encoder=config.encoder,
                        cache=config.cache, paths=PathsConfig(
                            metadata_dir=str(output / "confirmation"), embeddings_dir=str(output / "confirmation")))
    encoder = Wav2Vec2Encoder(checkpoint=config.encoder.checkpoint, pooling=config.encoder.pooling,
                              normalize=config.encoder.normalize, device=config.encoder.device,
                              batch_size=config.encoder.batch_size)
    actual = asdict(encoder.info)
    # Device/batch scheduling may change; encoder semantics/revision may not.
    for key in ["name", "checkpoint", "embedding_dim", "pooling", "l2_normalized", "revision"]:
        if actual[key] != expected_encoder[key]:
            raise ValueError(f"Confirmation encoder differs in {key}")
    feature_cache = CacheStore(Path(config.cache.path) / "features")
    embedding_cache = CacheStore(Path(config.cache.path) / "embeddings")
    try:
        rows, matrix, ids, _ = process_split(records, "validation", config, feature_cache,
                                            embedding_cache, encoder, norm_stats)
    finally:
        feature_cache.close()
        embedding_cache.close()
    directory = output / "confirmation"
    save_metadata_parquet(rows, directory / "metadata.parquet")
    save_embedding_matrix(matrix, ids, directory / "confirmation.npy", directory / "index.parquet")
    # build_metadata_row follows config.paths; record the actual persisted path.
    frame = pd.DataFrame(rows)
    frame["embedding_path"] = str(directory / "confirmation.npy")
    frame.to_parquet(directory / "metadata.parquet", index=False)
    assert_disjoint(train, frame)
    assert_disjoint(tuning, frame)
    (directory / "manifest.json").write_text(json.dumps({
        "role": "fresh confirmation on unused official-validation speakers", "encoder": actual,
        "samples": len(frame), "speakers": int(frame.speaker_id.nunique()),
        "source_fingerprint": source_fingerprint(),
        "artifacts": {str(p.resolve()): file_content_hash(p) for p in
                      [directory / "metadata.parquet", directory / "confirmation.npy", directory / "index.parquet"]},
    }, indent=2))
    return frame, matrix


def cases(strategy, alphas, fractions):
    if strategy == "combined":
        return [(f"alpha={a:g}", lambda a=a: CombinedSelector(a)) for a in sorted(set(alphas))]
    if strategy == "quality_representative":
        return [(f"retain={f:g}", lambda f=f: QualityRepresentativeSelector(f)) for f in sorted(set(fractions))]
    return [("default", {"random": RandomSelector, "quality": QualitySelector,
                          "diversity": DiversitySelector, "representative": RepresentativeSelector}[strategy])]


def run_study(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists() or (output / "locked_choices.json").exists():
        raise ValueError("Use a new output directory for a new study")
    if (not args.seeds or len(args.seeds) != len(set(args.seeds)) or any(b <= 0 for b in args.budgets)
            or any(not np.isfinite(c) or c <= 0 for c in args.c_values)
            or any(not 0 <= a <= 1 for a in args.alphas)
            or any(not 0 < f < 1 for f in args.retain_fractions)):
        raise ValueError("Invalid study settings")
    config = AuraConfig.from_yaml(args.config)
    train, x_train, train_manifest = load_split(config, "train")
    tuning, x_tuning, tuning_manifest = load_split(config, "validation")
    assert_disjoint(train, tuning)
    if train_manifest["encoder"] != tuning_manifest["encoder"] or any(b > len(train) for b in args.budgets):
        raise ValueError("Incompatible encoders or budgets")
    if set(train.label) != set(tuning.label):
        raise ValueError("Tuning labels differ from training")
    (output / "protocol.json").write_text(json.dumps({
        "arguments": vars(args), "confirmation_rule": "all official-validation speakers absent from train and existing tuning",
        "selection_rule": "mean tuning accuracy, then macro-F1, then lexical setting and lower C",
        "previous_test_set": "not opened or used in this study",
        "script_hash": file_content_hash(__file__),
        "selector_hash": file_content_hash(Path(__file__).resolve().parent.parent / "experiments/representative.py"),
    }, indent=2))
    embeddings = dict(zip(train.id, x_train))
    row_for_id = dict(zip(train.id, range(len(train))))
    labels = sorted(train.label.unique())
    selections_dir = output / "selections"
    selections_dir.mkdir()
    sweep, finalists, locks = [], [], []
    jobs = [(s, b) for b in sorted(set(args.budgets)) for s in
            ["random", "quality", "diversity", "combined", "representative", "quality_representative"]]
    jobs += [("full_pool", len(train))]
    for strategy, budget in jobs:
        configurations = [("default", None)] if strategy == "full_pool" else cases(strategy, args.alphas, args.retain_fractions)
        candidates = []
        for setting, factory in configurations:
            seeds = args.seeds[:1] if strategy == "full_pool" else args.seeds
            for seed in seeds:
                t0 = time.perf_counter()
                if factory is None:
                    ids = train.id.tolist()
                    record = {"selected_ids": ids, "strategy": strategy, "seed": seed}
                else:
                    result = factory().select(train, embeddings, budget, seed)
                    ids, record = result.selected_ids, result.to_dict()
                selection_seconds = time.perf_counter() - t0
                selection_path = selections_dir / f"{strategy}_B{budget}_seed{seed}_{setting.replace('=', '-')}.json"
                selection_path.write_text(json.dumps(record, indent=2))
                indices = [row_for_id[sid] for sid in ids]
                geometry = coverage_diagnostics(train, embeddings, ids)
                for c in sorted(set(args.c_values)):
                    model, fit_seconds = fit_probe(x_train[indices], train.label.to_numpy()[indices], c)
                    scores = metrics(tuning.label, model.predict(x_tuning), labels)
                    row = {"strategy": strategy, "budget": budget, "seed": seed, "setting": setting,
                           "C": c, "validation_accuracy": scores["accuracy"], "validation_macro_f1": scores["macro_f1"],
                           "selection_seconds": selection_seconds, "fit_seconds": fit_seconds,
                           "mean_quality": float(train.iloc[indices].quality_score.mean()),
                           "distinct_speakers": int(train.iloc[indices].speaker_id.nunique()),
                           "selection_path": str(selection_path), **geometry}
                    sweep.append(row)
                    candidates.append((row, model))
        rank = pd.DataFrame([r for r, _ in candidates]).groupby(["setting", "C"], as_index=False)[
            ["validation_accuracy", "validation_macro_f1"]].mean()
        best = rank.sort_values(["validation_accuracy", "validation_macro_f1", "setting", "C"],
                                 ascending=[False, False, True, True]).iloc[0]
        chosen = [(r, model) for r, model in candidates if r["setting"] == best.setting and r["C"] == best.C]
        finalists.extend(chosen)
        locks.append({"strategy": strategy, "budget": budget, "setting": best.setting, "C": float(best.C),
                      "mean_validation_accuracy": float(best.validation_accuracy)})
        pd.DataFrame(sweep).to_csv(output / "validation_sweep.csv", index=False)
        print(f"Locked {strategy} B={budget}: {best.setting}, C={best.C:g}, validation={best.validation_accuracy:.4f}", flush=True)
    (output / "locked_choices.json").write_text(json.dumps(locks, indent=2))
    confirmation, x_confirmation = prepare_confirmation(config, train, tuning, output, train_manifest["encoder"])
    assert_disjoint(train, confirmation)
    assert_disjoint(tuning, confirmation)
    if set(confirmation.label) != set(labels):
        raise ValueError("Confirmation labels differ from training")
    runs, per_class = [], []
    for record, model in finalists:
        predictions = model.predict(x_confirmation)
        scores = metrics(confirmation.label, predictions, labels)
        runs.append({**record, "confirmation_accuracy": scores["accuracy"], "confirmation_macro_f1": scores["macro_f1"]})
        per_class.append({"strategy": record["strategy"], "budget": record["budget"], "seed": record["seed"],
                          "recall": dict(zip(labels, recall_score(confirmation.label, predictions,
                              labels=labels, average=None, zero_division=0).tolist())),
                          "confusion_matrix": confusion_matrix(confirmation.label, predictions, labels=labels).tolist()})
    df = pd.DataFrame(runs)
    df.to_csv(output / "runs.csv", index=False)
    summary = df.groupby(["strategy", "budget"], as_index=False).agg(
        seeds=("seed", "count"), validation_accuracy=("validation_accuracy", "mean"),
        confirmation_accuracy=("confirmation_accuracy", "mean"), confirmation_std=("confirmation_accuracy", "std"),
        confirmation_macro_f1=("confirmation_macro_f1", "mean"), mean_quality=("mean_quality", "mean"),
        mean_coverage_distance=("mean_coverage_distance", "mean"), selected_knn_distance=("selected_knn_distance", "mean"),
        distinct_speakers=("distinct_speakers", "mean"), selection_seconds=("selection_seconds", "mean"),
        fit_seconds=("fit_seconds", "mean"), setting=("setting", "first"), C=("C", "first"))
    summary.to_csv(output / "summary.csv", index=False)
    random = df[df.strategy == "random"][["budget", "seed", "confirmation_accuracy"]].rename(
        columns={"confirmation_accuracy": "random_accuracy"})
    paired = df[df.strategy != "full_pool"].merge(random, on=["budget", "seed"])
    paired["delta_pp"] = 100 * (paired.confirmation_accuracy - paired.random_accuracy)
    paired.to_csv(output / "paired_differences.csv", index=False)
    (output / "per_class.json").write_text(json.dumps({"labels": labels, "runs": per_class}, indent=2))
    (output / "experiment.json").write_text(json.dumps({
        "config": asdict(config), "arguments": vars(args), "source_fingerprint": source_fingerprint(),
        "script_hash": file_content_hash(__file__), "runtime_versions": runtime_versions(),
        "sklearn_version": sklearn.__version__, "elapsed_seconds": time.perf_counter() - started,
        "train_manifest": train_manifest, "tuning_manifest": tuning_manifest,
        "confirmation_samples": len(confirmation), "confirmation_speakers": int(confirmation.speaker_id.nunique()),
        "experimental_source_hashes": {str(p): file_content_hash(p) for p in
            [Path(__file__).resolve().parent.parent / "experiments/representative.py",
             Path(__file__).resolve().parent / "evaluate_selection.py"]},
        "test_data_accessed": False,
    }, indent=2))
    lines = ["# Representative selection study", "",
             f"Training candidates: {len(train)}. Tuning: {len(tuning)}. Fresh confirmation: "
             f"{len(confirmation)} recordings from {confirmation.speaker_id.nunique()} unused speakers.", "",
             "All settings were locked using tuning validation before confirmation preparation. "
             "The previously inspected test set was not opened. This small confirmation set is not comparable "
             "directly to the previous 4,074-recording test set.", "",
             "| Strategy | Budget | Tuning accuracy | Confirmation accuracy ± seed std | Setting |",
             "|---|---:|---:|---:|---|"]
    for r in summary.itertuples():
        std = f"{100*r.confirmation_std:.2f} pp" if np.isfinite(r.confirmation_std) else "n/a"
        lines.append(f"| {r.strategy} | {r.budget} | {100*r.validation_accuracy:.2f}% | "
                     f"{100*r.confirmation_accuracy:.2f}% ± {std} | {r.setting}; C={r.C:g} |")
    lines += ["", "Geometry diagnostics are computed only from the training pool. Lower average coverage "
              "distance means the pool is nearer to selected representatives. Higher selected kNN distance "
              "means more isolated selections. Neither is a substitute for classifier performance.", "",
              "Variation across selection seeds is not uncertainty over new datasets. Deterministic algorithms "
              "can yield identical subsets and zero seed variation. Development used one small candidate pool "
              "and one previously used tuning set. No settings were adjusted after confirmation scores appeared."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.v2.yaml")
    parser.add_argument("--output-dir", default="results/representative_v3")
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 200, 400, 800])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1, 10])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--retain-fractions", nargs="+", type=float, default=[0.5, 0.75, 0.9])
    run_study(parser.parse_args())


if __name__ == "__main__":
    main()
