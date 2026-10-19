#!/usr/bin/env python
"""Parallel execution of the already-declared scaling jobs, training data only.

Stop the sequential benchmark before using this controller. It writes the
same checkpoint identity/schema that the frozen benchmark verifies on resume.
Only execution scheduling changes; settings, seeds, and algorithms do not.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from aura.cache.store import file_content_hash, config_fingerprint
from aura.provenance import source_fingerprint
from experiments.scaling import load_scaling_artifact
from scripts.benchmark_scaling_study import make_selector
from scripts.evaluate_selection import fit_probe

STATE = {}


def initialize(data_root, output, workers):
    root, output = Path(data_root), Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    frame, matrix, _ = load_scaling_artifact(root, "train")
    STATE.update(root=root, output=output, protocol=protocol, frame=frame, matrix=matrix,
                 embeddings=dict(zip(frame.id, matrix)), rows=dict(zip(frame.id, range(len(frame)))),
                 workers=workers, protocol_hash=file_content_hash(output / "protocol.json"),
                 train_hash=file_content_hash(root / "metadata/train_manifest.json"))


def fit_job(job, seed):
    state = STATE
    frame, matrix = state["frame"], state["matrix"]
    strategy, budget = job["strategy"], job["budget"]
    directory = state["output"] / "jobs" / f"{strategy}_B{budget}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = config_fingerprint({"job": job, "seed": seed, "protocol": state["protocol_hash"],
                                   "train_manifest": state["train_hash"]})
    started = time.perf_counter()
    if strategy == "full_pool":
        ids = frame.id.tolist()
        selection = {"strategy": strategy, "selected_ids": ids, "seed": seed}
    else:
        selected = make_selector(strategy, job["setting"]).select(frame, state["embeddings"], budget, seed)
        ids, selection = selected.selected_ids, selected.to_dict()
    selection_seconds = time.perf_counter() - started
    rows = [state["rows"][sid] for sid in ids]
    model, fit_seconds = fit_probe(matrix[rows], frame.label.to_numpy()[rows], job["C"])
    selection_path = directory / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2))
    temporary_model = directory / "model.partial.joblib"
    joblib.dump(model, temporary_model)
    temporary_model.replace(directory / "model.joblib")
    record = {**job, "seed": seed, "identity": identity, "selection_seconds": selection_seconds,
              "fit_seconds": fit_seconds, "distinct_speakers": int(frame.iloc[rows].speaker_id.nunique()),
              "mean_quality": float(frame.iloc[rows].quality_score.mean()), "selection_path": str(selection_path),
              "parallel_workers": state["workers"], "worker_pid": os.getpid(),
              "worker_script_hash": file_content_hash(__file__),
              "artifacts": {str(p.resolve()): file_content_hash(p) for p in
                            [selection_path, directory / "model.joblib"]}}
    temporary_manifest = directory / "manifest.partial.json"
    temporary_manifest.write_text(json.dumps(record, indent=2))
    temporary_manifest.replace(directory / "manifest.json")
    return {"strategy": strategy, "budget": budget, "seed": seed,
            "selection_seconds": selection_seconds, "fit_seconds": fit_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/scaling_v4")
    parser.add_argument("--output-dir", default="results/scaling_v4")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    root, output = Path(args.data_root), Path(args.output_dir)
    if (output / "models_locked.json").exists() or (output / "experiment.json").exists():
        raise ValueError("Models are already locked or results observed")
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    benchmark = Path(__file__).resolve().parent / "benchmark_scaling_study.py"
    if (protocol["script_hash"] != file_content_hash(benchmark)
            or protocol["source_fingerprint"] != source_fingerprint()):
        raise ValueError("Frozen benchmark code changed")
    lock = output / "parallel_controller.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    started = time.perf_counter()
    completed, pending, results = [], [], []
    try:
        train_hash = file_content_hash(root / "metadata/train_manifest.json")
        for job in protocol["jobs"]:
            seeds = protocol["arguments"]["seeds"][:1] if job["strategy"] == "full_pool" else protocol["arguments"]["seeds"]
            for seed in seeds:
                directory = output / "jobs" / f"{job['strategy']}_B{job['budget']}_seed{seed}"
                path = directory / "manifest.json"
                if path.exists():
                    try:
                        record = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        path.replace(directory / "manifest.interrupted.json")
                    else:
                        expected = config_fingerprint({"job": job, "seed": seed,
                            "protocol": file_content_hash(protocol_path), "train_manifest": train_hash})
                        if record["identity"] != expected:
                            raise ValueError("Checkpoint identity differs from protocol")
                        for artifact, digest in record["artifacts"].items():
                            if file_content_hash(artifact) != digest:
                                raise ValueError(f"Changed checkpoint artifact: {artifact}")
                        completed.append(str(path))
                        continue
                pending.append((job, seed))
        print(f"Reusing {len(completed)} checkpoints; executing {len(pending)} jobs with {args.workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(root), str(output), args.workers)) as pool:
            futures = [pool.submit(fit_job, job, seed) for job, seed in pending]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"Parallel ready {result['strategy']} B={result['budget']} seed={result['seed']}: "
                      f"select={result['selection_seconds']:.2f}s fit={result['fit_seconds']:.2f}s "
                      f"({len(completed)+len(results)}/{len(completed)+len(pending)})", flush=True)
        (output / "parallel_execution.json").write_text(json.dumps({
            "arguments": vars(args), "controller_hash": file_content_hash(__file__),
            "protocol_hash": file_content_hash(protocol_path), "reused_serial_checkpoints": completed,
            "new_jobs": results, "workers": args.workers, "threads_per_worker": 2,
            "wall_seconds": time.perf_counter() - started,
            "scope": "training-only execution scheduling; no evaluation data or parameter changes",
            "timing_limit": "concurrent wall times include contention and cannot be treated as isolated algorithm benchmarks",
        }, indent=2))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
