"""Validate frozen declarations, model checkpoints and saved evaluation rows."""
import argparse
from collections import Counter
import json
from importlib.metadata import version
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from aura.cache.store import file_content_hash, config_fingerprint
from aura.provenance import source_fingerprint
from experiments.scaling import load_scaling_artifact
from scripts.evaluate_selection import assert_disjoint, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    root = Path("data/scaling_v4" if args.control else "data/audiomnist_v5")
    out = Path("results/duplicate_control_v5" if args.control else "results/audiomnist_v5")
    protocol = json.loads((out / "protocol.json").read_text())
    plan = json.loads((root / "partition.json").read_text())
    if protocol["source_fingerprint"] != source_fingerprint():
        raise ValueError("Core source differs")
    if protocol["partition_hash"] != file_content_hash(root / "partition.json"):
        raise ValueError("Partition differs")
    for path, expected in protocol["sources"].items():
        if file_content_hash(path) != expected:
            raise ValueError(f"Experimental source differs: {path}")
    print("Verifying raw content hashes", flush=True)
    for record in plan["train"] + plan["holdout"]:
        if file_content_hash(record["audio_path"]) != record["audio_hash"]:
            raise ValueError("Raw audio differs from declaration")
    train, _, train_manifest = load_scaling_artifact(root, "train")
    holdout, _, holdout_manifest = load_scaling_artifact(root, "holdout")
    if train_manifest["encoder"] != holdout_manifest["encoder"]:
        raise ValueError("Training and holdout encoders differ")
    assert_disjoint(train, holdout)
    if train.id.tolist() != [r["sample_id"] for r in plan["train"]] or holdout.id.tolist() != [r["sample_id"] for r in plan["holdout"]]:
        raise ValueError("Prepared records differ from declaration")
    if {r["audio_hash"] for r in plan["train"]} & {r["audio_hash"] for r in plan["holdout"]}:
        raise ValueError("Content overlap")
    lock = json.loads((out / "models_locked.json").read_text())
    if len(lock) != len(protocol["jobs"]):
        raise ValueError("Locked model count differs")
    signature = config_fingerprint(dict(protocol=file_content_hash(out / "protocol.json"),
                                       training=file_content_hash(root / "metadata/train_manifest.json")))
    expected_jobs = {json.dumps(job, sort_keys=True) for job in protocol["jobs"]}
    actual_jobs = set()
    for path, expected in lock.items():
        if file_content_hash(path) != expected:
            raise ValueError("Model manifest differs")
        saved = json.loads(Path(path).read_text())
        job = saved["job"]
        if saved["identity"] != config_fingerprint(dict(job=job, protocol=signature)):
            raise ValueError("Model identity differs")
        actual_jobs.add(json.dumps(job, sort_keys=True))
        for artifact, digest in saved["artifacts"].items():
            if file_content_hash(artifact) != digest:
                raise ValueError("Model artifact differs")
        ids = json.loads((Path(path).parent / "selection.json").read_text())
        if len(ids) != job["budget"] or len(set(ids)) != len(ids) or not set(ids) <= set(train.id):
            raise ValueError("Invalid selection")
        if job["strategy"] != "full_pool":
            counts = train.set_index("id").loc[ids].label.value_counts()
            if len(counts) != train.label.nunique() or counts.max() - counts.min() > 1:
                raise ValueError("Selection is not class balanced")
    if expected_jobs != actual_jobs:
        raise ValueError("Locked jobs differ from protocol")
    runs = pd.read_csv(out / "runs.csv")
    preds = pd.read_csv(out / "predictions.csv.gz", dtype={"label": str, "prediction": str})
    if len(runs) != len(lock) or len(preds) != len(holdout) * len(runs):
        raise ValueError("Incomplete evaluation")
    for row in runs.itertuples():
        group = preds[(preds.strategy == row.strategy) & (preds.budget == row.budget) & (preds.seed == row.seed)]
        if group.id.tolist() != holdout.id.tolist() or not group.label.tolist() == holdout.label.astype(str).tolist():
            raise ValueError("Prediction cohort differs")
        scores = metrics(group.label, group.prediction, sorted(train.label.unique()))
        if not np.isclose(scores["accuracy"], row.accuracy, atol=1e-12) or not np.isclose(scores["macro_f1"], row.macro_f1, atol=1e-12):
            raise ValueError("Saved metrics differ from predictions")
    counts = Counter(r["audio_hash"] for r in plan["train"])
    payload = dict(all_checks_passed=True, models=len(lock), training_records=len(train),
        holdout_records=len(holdout), training_speakers=int(train.speaker_id.nunique()),
        holdout_speakers=int(holdout.speaker_id.nunique()), redundant_training_copies=len(train)-len(counts),
        raw_hashes_checked=len(plan["train"])+len(plan["holdout"]),
        training_preparation_seconds=train_manifest["elapsed_seconds"],
        python=sys.version, packages={name: version(name) for name in
            ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "torch", "torchaudio", "transformers", "librosa", "soundfile"]},
        script_hash=file_content_hash(__file__), protocol_hash=file_content_hash(out / "protocol.json"))
    (out / "verification.json").write_text(json.dumps(payload, indent=2))
    complete = dict(status="complete", verification=payload,
        artifacts={str(path.resolve()): file_content_hash(path) for path in out.iterdir()
                   if path.is_file() and path.name != "experiment.json"},
        data_manifests={str(path.resolve()): file_content_hash(path) for path in
                        [root / "metadata/train_manifest.json", root / "metadata/holdout_manifest.json"]})
    (out / "experiment.json").write_text(json.dumps(complete, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
