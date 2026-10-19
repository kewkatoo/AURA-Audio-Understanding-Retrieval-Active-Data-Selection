#!/usr/bin/env python
"""Frozen settings, a large candidate pool, and untouched training speakers.

Settings transfer from the v3 tuning choices. For a new larger budget, the
largest previously tuned budget's settings are used. No new parameter search
or previously inspected validation/test score is used to choose models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from aura.cache.store import file_content_hash, config_fingerprint
from aura.config import AuraConfig
from aura.provenance import source_fingerprint, runtime_versions
from aura.selection.combined import CombinedSelector
from aura.selection.diversity import DiversitySelector
from aura.selection.quality import QualitySelector
from aura.selection.random import RandomSelector
from experiments.representative import RepresentativeSelector, QualityRepresentativeSelector
from experiments.scaling import load_scaling_artifact
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap
from scripts.evaluate_selection import assert_disjoint, fit_probe, metrics
from scripts.prepare_scaling_study import prepare_role


def transferred_settings(choices, strategy, budget):
    eligible = [r for r in choices if r["strategy"] == strategy and r["budget"] <= budget]
    if not eligible:
        raise ValueError(f"No prior tuning choice for {strategy} at budget {budget}")
    choice = max(eligible, key=lambda r: r["budget"])
    return {"strategy": strategy, "budget": budget, "setting": choice["setting"],
            "C": float(choice["C"]), "source_budget": choice["budget"]}


def make_selector(strategy, setting):
    if strategy == "combined":
        return CombinedSelector(float(setting.split("=")[1]))
    if strategy == "quality_representative":
        return QualityRepresentativeSelector(float(setting.split("=")[1]))
    return {"random": RandomSelector, "quality": QualitySelector, "diversity": DiversitySelector,
            "representative": RepresentativeSelector}[strategy]()


def declare_protocol(args):
    root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    partition_path = root / "partition.json"
    plan = json.loads(partition_path.read_text())
    choices = json.loads(Path(args.prior_choices).read_text())
    prior = json.loads((Path(args.prior_choices).parent / "experiment.json").read_text())
    selector_path = Path(__file__).resolve().parent.parent / "experiments/representative.py"
    expected = prior["experimental_source_hashes"][str(selector_path)]
    if expected != file_content_hash(selector_path) or prior["source_fingerprint"] != source_fingerprint():
        raise ValueError("Frozen selectors or preparation code differ from the v3 study")
    if (len(set(args.seeds)) != len(args.seeds) or not args.seeds or "random" not in args.strategies
            or len(set(args.strategies)) != len(args.strategies)
            or any(b < 1 or b > len(plan["train"]) for b in args.budgets)):
        raise ValueError("Invalid seeds or budgets")
    jobs = [transferred_settings(choices, strategy, budget)
            for budget in sorted(set(args.budgets)) for strategy in args.strategies]
    full = next(r for r in choices if r["strategy"] == "full_pool")
    jobs.append({"strategy": "full_pool", "budget": len(plan["train"]), "setting": "default",
                 "C": float(full["C"]), "source_budget": full["budget"]})
    protocol = {"partition_hash": file_content_hash(partition_path), "arguments": vars(args), "jobs": jobs,
                "source_fingerprint": source_fingerprint(), "selector_hash": expected,
                "prior_choices_hash": file_content_hash(args.prior_choices), "script_hash": file_content_hash(__file__),
                "parameter_rule": "transfer prior tuning settings; largest lower/equal budget for new budgets",
                "holdout_rule": f"{plan['held_out_speakers']} randomly chosen previously unseen official-training speakers; content duplicates excluded",
                "no_new_tuning": True, "previous_validation_or_test_loaded": False}
    # declare-only changes only execution control, not the statistical protocol.
    protocol["arguments"] = {k: v for k, v in vars(args).items() if k != "declare_only"}
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Protocol differs; use a new results directory")
    else:
        if (root / "metadata/holdout_manifest.json").exists():
            raise ValueError("Cannot declare a new protocol after holdout preparation")
        path.write_text(json.dumps(protocol, indent=2))
    return plan, protocol


def run(args):
    started = time.perf_counter()
    root, output = Path(args.data_root), Path(args.output_dir)
    plan, protocol = declare_protocol(args)
    if args.declare_only:
        print(f"Locked {len(protocol['jobs'])} method/budget settings before holdout preparation", flush=True)
        return
    if (output / "experiment.json").exists():
        raise ValueError("Benchmark is complete; do not overwrite an observed result")
    train, matrix, train_manifest = load_scaling_artifact(root, "train")
    embeddings = dict(zip(train.id, matrix))
    row_for_id = dict(zip(train.id, range(len(train))))
    labels = sorted(train.label.unique())
    finalists = []
    for job in protocol["jobs"]:
        strategy, budget = job["strategy"], job["budget"]
        seeds = args.seeds[:1] if strategy == "full_pool" else args.seeds
        for seed in seeds:
            directory = output / "jobs" / f"{strategy}_B{budget}_seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            identity = config_fingerprint({"job": job, "seed": seed, "protocol": file_content_hash(output / "protocol.json"),
                                           "train_manifest": file_content_hash(root / "metadata/train_manifest.json")})
            if manifest_path.exists():
                record = json.loads(manifest_path.read_text())
                if record["identity"] != identity:
                    raise ValueError("Checkpoint belongs to a different protocol")
                for path, expected in record["artifacts"].items():
                    if file_content_hash(path) != expected:
                        raise ValueError(f"Changed job artifact: {path}")
                model = joblib.load(directory / "model.joblib")
            else:
                t0 = time.perf_counter()
                if strategy == "full_pool":
                    ids = train.id.tolist()
                    selection = {"strategy": strategy, "selected_ids": ids, "seed": seed}
                else:
                    selected = make_selector(strategy, job["setting"]).select(train, embeddings, budget, seed)
                    ids, selection = selected.selected_ids, selected.to_dict()
                selection_seconds = time.perf_counter() - t0
                rows = [row_for_id[sid] for sid in ids]
                model, fit_seconds = fit_probe(matrix[rows], train.label.to_numpy()[rows], job["C"])
                selection_path = directory / "selection.json"
                selection_path.write_text(json.dumps(selection, indent=2))
                joblib.dump(model, directory / "model.joblib")
                record = {**job, "seed": seed, "identity": identity, "selection_seconds": selection_seconds,
                          "fit_seconds": fit_seconds, "distinct_speakers": int(train.iloc[rows].speaker_id.nunique()),
                          "mean_quality": float(train.iloc[rows].quality_score.mean()),
                          "selection_path": str(selection_path),
                          "artifacts": {str(p.resolve()): file_content_hash(p) for p in
                                        [selection_path, directory / "model.joblib"]}}
                manifest_path.write_text(json.dumps(record, indent=2))
            finalists.append((record, model))
            print(f"Ready {strategy} B={budget} seed={seed}: select={record['selection_seconds']:.2f}s, fit={record['fit_seconds']:.2f}s", flush=True)
    (output / "models_locked.json").write_text(json.dumps({"jobs": len(finalists),
        "protocol_hash": file_content_hash(output / "protocol.json"),
        "checkpoint_hashes": {str(p): file_content_hash(p) for p in sorted((output / "jobs").glob("*/manifest.json"))}}, indent=2))
    # No new configuration choices occur after this point.
    config = AuraConfig.from_yaml(args.config)
    prepare_role(config, root, plan, "holdout", args.chunk_size, output / "protocol.json")
    holdout, x_holdout, holdout_manifest = load_scaling_artifact(root, "holdout")
    assert_disjoint(train, holdout)
    if train_manifest["encoder"] != holdout_manifest["encoder"]:
        raise ValueError("Encoder differs between candidate pool and holdout")
    if set(holdout.label) != set(labels):
        raise ValueError("Holdout class coverage differs")
    y_holdout = holdout.label.to_numpy()
    runs, predictions, correctness = [], [], {}
    for record, model in finalists:
        guessed = model.predict(x_holdout)
        scores = metrics(y_holdout, guessed, labels)
        values = (guessed == y_holdout).astype(float)
        correctness[(record["strategy"], record["budget"], record["seed"])] = values
        runs.append({k: v for k, v in dict(record, holdout_accuracy=scores["accuracy"],
                     holdout_macro_f1=scores["macro_f1"]).items() if k not in {"artifacts", "identity"}})
        predictions.extend({"strategy": record["strategy"], "budget": record["budget"], "seed": record["seed"],
                            "id": sid, "speaker_id": speaker, "label": target, "prediction": prediction,
                            "correct": int(correct)} for sid, speaker, target, prediction, correct in
                           zip(holdout.id, holdout.speaker_id, y_holdout, guessed, values))
    frame = pd.DataFrame(runs)
    frame.to_csv(output / "runs.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    summary = frame.groupby(["strategy", "budget"], as_index=False).agg(
        seeds=("seed", "count"), accuracy=("holdout_accuracy", "mean"), std=("holdout_accuracy", "std"),
        macro_f1=("holdout_macro_f1", "mean"), selection_seconds=("selection_seconds", "mean"),
        fit_seconds=("fit_seconds", "mean"), distinct_speakers=("distinct_speakers", "mean"),
        setting=("setting", "first"), C=("C", "first"))
    summary.to_csv(output / "summary.csv", index=False)
    differences = []
    for (strategy, budget), group in frame[frame.strategy != "full_pool"].groupby(["strategy", "budget"]):
        a = np.mean([correctness[(strategy, budget, seed)] for seed in group.seed], axis=0)
        b = np.mean([correctness[("random", budget, seed)] for seed in group.seed], axis=0)
        low, high = speaker_bootstrap(a - b, holdout.speaker_id, 2000, 0)
        differences.append({"strategy": strategy, "budget": int(budget), "delta_pp": float((a-b).mean()*100),
                            "bootstrap_low_pp": low, "bootstrap_high_pp": high})
    pd.DataFrame(differences).to_csv(output / "speaker_bootstrap.csv", index=False)
    (output / "experiment.json").write_text(json.dumps({
        "protocol": protocol, "train_manifest": train_manifest, "holdout_manifest": holdout_manifest,
        "runtime_versions": runtime_versions(), "threadpools": threadpool_info(),
        "elapsed_seconds_this_invocation": time.perf_counter()-started,
        "all_candidate_count": len(train), "holdout_count": len(holdout),
        "held_out_speakers": int(holdout.speaker_id.nunique()), "previous_validation_or_test_loaded": False,
        "parameter_search_performed": False,
    }, indent=2))
    lines = ["# Full-corpus scaling benchmark", "",
             f"Available official-training recordings: {plan['available_recordings']}. "
             f"Candidate pool: {len(train)}. Fresh speaker holdout: {len(holdout)} recordings from "
             f"{holdout.speaker_id.nunique()} speakers. Duplicate exclusions: {len(plan['excluded_duplicates'])}.", "",
             "All selection/head settings transferred from the earlier v3 tuning choices. New larger budgets "
             "use the largest earlier budget's settings. No parameter search was performed. Prior evaluation "
             "sets were not loaded. The full-pool baseline excludes this newly reserved holdout.", "",
             "| Strategy | Budget | Holdout accuracy ± seed std | Macro-F1 | Selection time |",
             "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        std = f"{100*row.std:.2f} pp" if np.isfinite(row.std) else "n/a"
        lines.append(f"| {row.strategy} | {row.budget} | {100*row.accuracy:.2f}% ± {std} | "
                     f"{row.macro_f1:.4f} | {row.selection_seconds:.2f}s |")
    lines += ["", "Speaker-bootstrap differences versus random are in speaker_bootstrap.csv. These exploratory "
              "intervals condition on this candidate pool and transferred settings and are not adjusted for "
              "multiple comparisons. Scores cannot be directly compared with earlier evaluation cohorts.", "",
              "This remains one dataset/task with a frozen embedding linear probe. It does not establish "
              "annotation savings, raw-audio fine-tuning savings, or independent-dataset generalization."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.v2.yaml")
    parser.add_argument("--data-root", default="data/scaling_v4")
    parser.add_argument("--output-dir", default="results/scaling_v4")
    parser.add_argument("--prior-choices", default="results/representative_v3/locked_choices.json")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 400, 800, 1600])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--strategies", nargs="+", choices=["random", "quality", "diversity", "combined", "representative", "quality_representative"],
                        default=["random", "quality", "diversity", "combined", "representative", "quality_representative"])
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--declare-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
