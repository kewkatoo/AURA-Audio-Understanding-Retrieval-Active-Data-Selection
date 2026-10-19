#!/usr/bin/env python
"""Frozen transfer to AudioMNIST and a post-hoc v4 duplicate control.

No validation tuning. All models are saved before holdout preparation/evaluation.
AudioMNIST uses a declared 45/15 speaker split, not its authors' benchmark split.
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from aura.cache.store import file_content_hash, config_fingerprint
from aura.config import AuraConfig
from aura.provenance import source_fingerprint, runtime_versions
from experiments.deduplicated import DeduplicatedRandomSelector
from experiments.scaling import partition_corpus, load_scaling_artifact
from scripts.prepare_scaling_study import prepare_role
from scripts.benchmark_scaling_study import transferred_settings, make_selector
from scripts.evaluate_selection import fit_probe, metrics, assert_disjoint
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap

REVISION = "630d7dab4c040882834de6fa21baf9a60372accd"
BUDGETS = [100, 200, 400, 800, 1600]
METHODS = ["random", "deduplicated_random", "representative", "quality_representative"]


def immutable_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Existing declaration differs: {path}")
    else:
        path.write_text(json.dumps(payload, indent=2))


def plan_audiomnist(root, repository):
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Dataset revision differs from declaration")
    config = AuraConfig.from_yaml("configs/experiment.v2.yaml")
    config = replace(config, dataset=replace(config.dataset, name="AudioMNIST", root=str(repository),
                                              classes=[str(i) for i in range(10)]))
    records = []
    for path in sorted((repository / "data").glob("*/*.wav")):
        digit, speaker, repetition = path.stem.split("_")
        records.append(dict(sample_id="audiomnist/" + path.stem, audio_path=str(path.resolve()),
                            label=digit, speaker_id="audiomnist/" + speaker,
                            utterance_number=int(repetition), split="train", audio_hash=file_content_hash(path)))
    if len(records) != 30000 or len({r["speaker_id"] for r in records}) != 60:
        raise ValueError("Unexpected AudioMNIST inventory")
    partition = partition_corpus(records, set(), set(), held_out_speakers=15, seed=20261018)
    plan = dict(config=asdict(config), source_fingerprint=source_fingerprint(), dataset_revision=revision,
                dataset_url="https://github.com/soerenab/AudioMNIST", license="MIT",
                license_hash=file_content_hash(repository / "LICENSE"), seed=20261018,
                available_recordings=len(records), **partition)
    immutable_json(root / "partition.json", plan)
    return config, plan


def declare(root, out, control=False):
    choices_path = Path("results/representative_v3/locked_choices.json")
    choices = json.loads(choices_path.read_text())
    methods = ["deduplicated_random"] if control else METHODS
    jobs = []
    for budget in BUDGETS:
        for method in methods:
            source = "random" if method == "deduplicated_random" else method
            job = transferred_settings(choices, source, budget)
            job["strategy"] = method
            jobs.extend([dict(job, seed=seed) for seed in range(5)])
    if not control:
        full = next(r for r in choices if r["strategy"] == "full_pool")
        jobs.append(dict(strategy="full_pool", budget=len(json.loads((root / "partition.json").read_text())["train"]),
                         seed=0, C=full["C"], setting="default"))
    sources = [Path(__file__), Path("experiments/deduplicated.py"), Path("experiments/representative.py"),
               Path("scripts/prepare_scaling_study.py"), Path("scripts/evaluate_selection.py"),
               Path("scripts/benchmark_scaling_study.py"), Path("experiments/scaling.py")]
    protocol = dict(partition_hash=file_content_hash(root / "partition.json"), source_fingerprint=source_fingerprint(),
                    sources={str(p.resolve()): file_content_hash(p) for p in sources},
                    choices_hash=file_content_hash(choices_path), runtime_versions=runtime_versions(), jobs=jobs,
                    purpose="post-hoc duplicate control on observed v4 holdout" if control else "independent dataset transfer",
                    tuning="none; settings transferred from Speech Commands v3; dedup uses random C",
                    primary_comparison="representative minus deduplicated_random at budget 100",
                    uncertainty="paired speaker bootstrap, fixed five selection seeds; exploratory, unadjusted")
    immutable_json(out / "protocol.json", protocol)
    return protocol


_TRAIN = None
def worker_init(root):
    global _TRAIN
    meta, matrix, manifest = load_scaling_artifact(root, "train")
    plan = json.loads((Path(root) / "partition.json").read_text())
    hashes = {r["sample_id"]: r["audio_hash"] for r in plan["train"]}
    meta["audio_hash"] = meta.id.map(hashes)
    _TRAIN = meta, matrix, dict(zip(meta.id, matrix)), dict(zip(meta.id, range(len(meta))))


def fit_job(payload):
    job, output, signature = payload
    meta, x, embeddings, positions = _TRAIN
    folder = Path(output) / "jobs" / f"{job['strategy']}_B{job['budget']}_seed{job['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = config_fingerprint(dict(job=job, protocol=signature))
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity"] != identity or any(file_content_hash(p) != h for p, h in saved["artifacts"].items()):
            raise ValueError("Changed model checkpoint")
        return str(folder)
    start = time.perf_counter()
    if job["strategy"] == "full_pool":
        ids = meta.id.tolist()
    else:
        selector = DeduplicatedRandomSelector() if job["strategy"] == "deduplicated_random" else make_selector(job["strategy"], job["setting"])
        ids = selector.select(meta, embeddings, job["budget"], job["seed"]).selected_ids
    selection_seconds = time.perf_counter() - start
    rows = [positions[sid] for sid in ids]
    model, seconds = fit_probe(x[rows], meta.label.to_numpy()[rows], job["C"])
    (folder / "selection.json").write_text(json.dumps(ids))
    joblib.dump(model, folder / "model.joblib")
    manifest_path.write_text(json.dumps(dict(identity=identity, job=job, selection_seconds=selection_seconds,
        fit_seconds=seconds, artifacts={str(p.resolve()): file_content_hash(p) for p in
            [folder / "selection.json", folder / "model.joblib"]}), indent=2))
    return str(folder)


def run(root, out, config, plan, control, workers):
    protocol = declare(root, out, control)
    if (out / "runs.csv").exists():
        print("Completed study already exists", flush=True)
        return
    prepare_role(config, root, plan, "train", 500, None)
    signature = config_fingerprint(dict(protocol=file_content_hash(out / "protocol.json"),
                                      training=file_content_hash(root / "metadata/train_manifest.json")))
    payloads = [(job, str(out), signature) for job in protocol["jobs"]]
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(str(root),)) as pool:
        folders = []
        for folder in pool.map(fit_job, payloads, chunksize=1):
            folders.append(Path(folder))
            print(f"Models ready: {len(folders)}/{len(payloads)} ({Path(folder).name})", flush=True)
    immutable_json(out / "models_locked.json", {str(p.resolve()): file_content_hash(p) for p in
                                               [f / "manifest.json" for f in folders]})
    prepare_role(config, root, plan, "holdout", 500, out / "protocol.json")
    train, _, _ = load_scaling_artifact(root, "train")
    holdout, x, _ = load_scaling_artifact(root, "holdout")
    assert_disjoint(train, holdout)
    if {r["audio_hash"] for r in plan["train"]} & {r["audio_hash"] for r in plan["holdout"]}:
        raise ValueError("Cross-role content overlap")
    runs, predictions = [], []
    labels = sorted(train.label.unique())
    for folder in folders:
        saved = json.loads((folder / "manifest.json").read_text())
        job = saved["job"]
        pred = joblib.load(folder / "model.joblib").predict(x)
        runs.append(dict(**job, **metrics(holdout.label, pred, labels),
                         selection_seconds=saved["selection_seconds"], fit_seconds=saved["fit_seconds"]))
        predictions.append(pd.DataFrame(dict(id=holdout.id, speaker_id=holdout.speaker_id,
            label=holdout.label, prediction=pred, strategy=job["strategy"], budget=job["budget"], seed=job["seed"])))
    frame = pd.DataFrame(runs)
    preds = pd.concat(predictions, ignore_index=True)
    frame.to_csv(out / "runs.csv", index=False)
    preds.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    frame.groupby(["strategy", "budget"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).to_csv(out / "summary.csv")
    if control:
        old = pd.read_csv("results/scaling_v4/predictions.csv.gz")
        # Existing prediction columns are inspected/normalized by report script.
        old.to_csv(out / "reference_predictions.csv.gz", index=False, compression="gzip")
    print(frame.groupby(["strategy", "budget"]).accuracy.mean().to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.control:
        root, out = Path("data/scaling_v4"), Path("results/duplicate_control_v5")
        plan = json.loads((root / "partition.json").read_text())
        config = AuraConfig.from_yaml("configs/experiment.v2.yaml")
    else:
        root, out = Path("data/audiomnist_v5"), Path("results/audiomnist_v5")
        config, plan = plan_audiomnist(root, Path("data/raw/AudioMNIST"))
    if args.declare_only:
        declare(root, out, args.control)
    else:
        run(root, out, config, plan, args.control, args.workers)


if __name__ == "__main__":
    main()
