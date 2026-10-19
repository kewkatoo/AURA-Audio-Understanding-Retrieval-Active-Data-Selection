#!/usr/bin/env python
"""Training-only corruption and speaker-held-out task-utility diagnostics.

These diagnostics assess the existing score; they do not fit new score weights.
Neither the tuning validation set nor any test set is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from aura.cache.store import file_content_hash
from aura.config import AuraConfig
from aura.ingest.audio import load_audio, to_mono, verify_and_resample, energy_based_vad
from aura.scoring.features import extract_features
from aura.scoring.quality import compute_quality_score
from scripts.evaluate_selection import load_split, fit_probe
from scripts.prepare_data import load_training_stats


def corruption_variants(waveform, rng):
    """Return explicit signal transforms; truncation is zero padding, not a label-preserving guarantee."""
    x = waveform.astype(np.float32)
    power = float(np.mean(x.astype(np.float64) ** 2))
    variants = [("clean", x.copy()), ("silence", np.zeros_like(x)), ("gain_0.25", x * 0.25)]
    noise = rng.normal(size=len(x)).astype(np.float32)
    noise /= max(float(np.sqrt(np.mean(noise ** 2))), 1e-12)
    for snr in [20, 10, 0]:
        variants.append((f"noise_{snr}db", x + noise * np.sqrt(power / 10 ** (snr / 10))))
    peak = float(np.max(np.abs(x)))
    reference = x / max(peak, 1e-12) * 0.5
    variants.append(("clip_reference", reference))
    for gain in [3, 8]:
        variants.append((f"clip_gain{gain}", np.clip(reference * gain, -1, 1)))
    for fraction in [0.5, 0.25]:
        shortened = x.copy()
        shortened[int(len(x) * fraction):] = 0
        variants.append((f"retain_prefix_{fraction:g}", shortened))
    return variants


def measured_features(waveform, config):
    activity = energy_based_vad(waveform, config.audio.sample_rate, config.audio.vad_frame_ms,
                               config.audio.vad_hop_ms, config.audio.vad_energy_percentile,
                               config.audio.vad_noise_margin_db, config.audio.vad_min_rms)
    return extract_features(waveform, config.audio.sample_rate, float(activity.mean()) if len(activity) else 0,
                            activity, config.audio.vad_frame_ms, config.audio.vad_hop_ms)


def run_diagnostics(args):
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        raise ValueError("Use a new output directory for a new diagnostic run")
    if args.per_class <= 0 or args.folds < 2:
        raise ValueError("Positive sample count and at least two folds are required")
    config = AuraConfig.from_yaml(args.config)
    train, matrix, manifest = load_split(config, "train")
    stats = load_training_stats(config)
    rng = np.random.default_rng(args.seed)
    # Fixed class-balanced sample from training only; selection precedes corruption.
    sampled = []
    for _, group in train.groupby("label"):
        ordered = group.sort_values("id")
        indices = rng.choice(len(ordered), min(args.per_class, len(ordered)), replace=False)
        sampled.extend(ordered.iloc[indices].to_dict("records"))
    records = []
    for row in sampled:
        waveform, sr = load_audio(row["audio_path"])
        waveform, _ = verify_and_resample(to_mono(waveform), sr, config.audio.sample_rate)
        for kind, transformed in corruption_variants(waveform, rng):
            features = measured_features(transformed, config)
            score = compute_quality_score(features, stats)
            records.append({"id": row["id"], "label": row["label"], "corruption": kind,
                            **{k: v for k, v in features.to_dict().items() if k not in {"mfcc_mean", "mfcc_std"}},
                            **score.to_dict()})
    corruption = pd.DataFrame(records)
    corruption.to_csv(output / "corruptions.csv", index=False)
    clean = corruption[corruption.corruption == "clean"][["id", "quality_score"]].rename(columns={"quality_score": "clean_quality"})
    corruption = corruption.merge(clean, on="id")
    corruption["quality_delta"] = corruption.quality_score - corruption.clean_quality
    summary = corruption.groupby("corruption", as_index=False).agg(
        examples=("id", "count"), mean_quality=("quality_score", "mean"),
        mean_delta_from_clean=("quality_delta", "mean"),
        fraction_lower_than_clean=("quality_delta", lambda x: float((x < -1e-6).mean())),
        mean_clipping=("clipping_ratio", "mean"), mean_activity=("speech_ratio", "mean"),
        mean_snr_proxy=("snr_db", "mean"))
    summary.to_csv(output / "corruption_summary.csv", index=False)
    # Clipping is compared with its gain-controlled reference, since peak
    # reference scaling itself can affect the activity detector's absolute floor.
    clipping = corruption[corruption.corruption.isin(["clip_reference", "clip_gain3", "clip_gain8"])].pivot(
        index="id", columns="corruption", values="quality_score")
    clipping.to_csv(output / "clipping_pairs.csv")
    # Predeclared fixed C=1; no tuning against these held-out speakers.
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    y = train.label.to_numpy()
    task_rows = []
    for fold, (fit, held_out) in enumerate(splitter.split(matrix, y, train.speaker_id)):
        if set(train.iloc[fit].speaker_id) & set(train.iloc[held_out].speaker_id):
            raise ValueError("Speaker leakage in task-utility folds")
        model, fit_seconds = fit_probe(matrix[fit], y[fit], 1)
        probabilities = model.predict_proba(matrix[held_out])
        class_index = {label: i for i, label in enumerate(model.classes_)}
        if any(label not in class_index for label in y[held_out]):
            raise ValueError("A fold is missing a training class")
        target_probability = probabilities[np.arange(len(held_out)), [class_index[label] for label in y[held_out]]]
        predictions = model.predict(matrix[held_out])
        for pos, p, predicted, target, true_p in zip(held_out, probabilities, predictions, y[held_out], target_probability):
            row = train.iloc[pos]
            task_rows.append({"id": row.id, "label": target, "speaker_id": row.speaker_id, "fold": fold,
                              "prediction": predicted, "correct": int(predicted == target),
                              "cross_entropy": float(-np.log(max(float(true_p), 1e-12))),
                              "quality_score": row.quality_score, "snr": row.snr,
                              "speech_ratio": row.speech_ratio, "clipping_ratio": row.clipping_ratio,
                              "rms_stability": row.rms_stability, "fit_seconds": fit_seconds})
        print(f"Speaker-held-out fold {fold + 1}/{args.folds} complete", flush=True)
    task = pd.DataFrame(task_rows)
    task.to_csv(output / "speaker_held_out_predictions.csv", index=False)
    correlations = []
    for column in ["quality_score", "snr", "speech_ratio", "clipping_ratio", "rms_stability"]:
        rho = float(spearmanr(task[column], task.cross_entropy).statistic)
        correlations.append({"feature": column, "spearman_with_loss": rho})
    pd.DataFrame(correlations).to_csv(output / "task_correlations.csv", index=False)
    task["quality_bin"] = pd.qcut(task.quality_score, 5, duplicates="drop")
    bins = task.groupby("quality_bin", observed=True).agg(
        examples=("id", "count"), quality_mean=("quality_score", "mean"),
        held_out_accuracy=("correct", "mean"), held_out_loss=("cross_entropy", "mean"))
    bins.to_csv(output / "quality_bins.csv")
    (output / "experiment.json").write_text(json.dumps({
        "arguments": vars(args), "training_manifest": manifest, "script_hash": file_content_hash(__file__),
        "elapsed_seconds": time.perf_counter() - started, "corruption_samples": len(sampled),
        "scope": "training-only diagnostics; fixed C=1 speaker-disjoint folds; no quality weights fitted",
        "validation_or_test_loaded": False,
    }, indent=2))
    lines = ["# Quality signal diagnostics", "",
             f"Controlled transforms on {len(sampled)} class-balanced training recordings; "
             f"{args.folds} speaker-disjoint classifier folds over {len(train)} training examples.", "",
             "| Transform | Mean quality | Change from clean | Fraction scoring lower than clean |",
             "|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.corruption} | {row.mean_quality:.4f} | {row.mean_delta_from_clean:+.4f} | "
                     f"{100*row.fraction_lower_than_clean:.1f}% |")
    lines += ["", "Clipping transforms have a separate peak-scaled clean reference; compare the paired scores "
              "in clipping_pairs.csv rather than attributing all changes from original clean audio to clipping.", "",
              "| Feature | Spearman correlation with held-out classifier loss |", "|---|---:|"]
    for row in correlations:
        lines.append(f"| {row['feature']} | {row['spearman_with_loss']:+.4f} |")
    lines += ["", "For quality, a negative correlation would support the hypothesis that higher quality "
              "predicts lower loss. These associations are descriptive, not causal or perceptual validation. "
              "No new quality weights were fitted. Truncation can remove the target word; it is a damage "
              "diagnostic, not a guaranteed label-preserving transform.", "",
              "The activity detector can respond to absolute gain and can miss noisy speech. Controlled "
              "noise is additive float audio, not an automatically clipped PCM recording. Neither tuning "
              "validation nor the existing test/confirmation sets were used."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.v2.yaml")
    parser.add_argument("--output-dir", default="results/quality_study_v3")
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    run_diagnostics(parser.parse_args())


if __name__ == "__main__":
    main()
