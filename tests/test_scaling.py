import numpy as np
import pytest

from experiments.scaling import partition_corpus


def test_scaling_partition_reserves_unseen_speakers_and_removes_content_leakage():
    rows = [dict(sample_id=f"s{s}-{c}", speaker_id=f"speaker{s}", label=c, audio_hash=f"h{s}-{c}")
            for s in range(8) for c in ["a", "b"]]
    first = partition_corpus(rows, {"speaker0", "speaker1"}, set(), 2, 4)
    assert not set(first["reserved_speakers"]) & {"speaker0", "speaker1"}
    assert first == partition_corpus(rows[::-1], {"speaker0", "speaker1"}, set(), 2, 4)
    held = first["holdout"][0]
    copied = dict(sample_id="duplicate", speaker_id="speaker0", label=held["label"], audio_hash=held["audio_hash"])
    result = partition_corpus(rows + [copied], {"speaker0", "speaker1"}, set(), 2, 4)
    assert copied in result["excluded_duplicates"]
    assert not {r["speaker_id"] for r in result["train"]} & {r["speaker_id"] for r in result["holdout"]}
    assert not {r["audio_hash"] for r in result["train"]} & {r["audio_hash"] for r in result["holdout"]}


def test_scaling_partition_excludes_previously_seen_content_from_holdout():
    rows = [dict(sample_id=f"s{s}-{c}", speaker_id=f"speaker{s}", label=c, audio_hash=f"h{s}-{c}")
            for s in range(8) for c in ["a", "b"]]
    first = partition_corpus(rows, {"speaker0"}, set(), 2, 9)
    known = first["holdout"][0]["audio_hash"]
    result = partition_corpus(rows, {"speaker0"}, {known}, 2, 9)
    assert not any(r["audio_hash"] == known for r in result["holdout"])
    with pytest.raises(ValueError):
        partition_corpus(rows, {r["speaker_id"] for r in rows}, set(), 1, 0)


def test_transferred_settings_do_not_search_new_budget():
    from scripts.benchmark_scaling_study import transferred_settings
    choices = [{"strategy": "quality_representative", "budget": 100, "setting": "retain=0.9", "C": 1},
               {"strategy": "quality_representative", "budget": 800, "setting": "retain=0.75", "C": 0.1}]
    result = transferred_settings(choices, "quality_representative", 1600)
    assert result == {"strategy": "quality_representative", "budget": 1600,
                      "source_budget": 800, "setting": "retain=0.75", "C": 0.1}
    with pytest.raises(ValueError):
        transferred_settings(choices, "quality_representative", 50)


def test_scaling_benchmark_fits_all_jobs_before_holdout(tmp_path, monkeypatch):
    from argparse import Namespace
    import json
    from pathlib import Path
    import pandas as pd
    from aura.cache.store import file_content_hash
    from aura.provenance import source_fingerprint
    from scripts import benchmark_scaling_study as study

    rng = np.random.default_rng(3)

    def dataset(split, n):
        labels = np.array(["a", "b"] * (n // 2))
        x = rng.normal(0, 0.02, (n, 4)).astype(np.float32)
        x[:, 0] += np.where(labels == "a", 1, -1)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        frame = pd.DataFrame({"id": [f"{split}-{i}" for i in range(n)], "label": labels,
                              "split": "train" if split == "train" else "validation",
                              "speaker_id": [f"{split}-{i}" for i in range(n)],
                              "quality_score": rng.uniform(size=n)})
        return frame, x, {"encoder": {"embedding_dim": 4}}

    train, holdout = dataset("train", 20), dataset("holdout", 10)
    root, out = tmp_path / "data", tmp_path / "results"
    (root / "metadata").mkdir(parents=True)
    (root / "metadata/train_manifest.json").write_text("{}")
    (root / "partition.json").write_text(json.dumps({"train": [{}] * 20, "holdout": [{}] * 10,
        "held_out_speakers": 10, "excluded_duplicates": [], "available_recordings": 30}))
    prior = tmp_path / "prior"
    prior.mkdir()
    choices = [{"strategy": s, "budget": 4 if s != "full_pool" else 20,
                "setting": "retain=0.75" if s == "quality_representative" else "default", "C": 1}
               for s in ["random", "quality_representative", "full_pool"]]
    (prior / "locked_choices.json").write_text(json.dumps(choices))
    selector = Path(study.__file__).resolve().parent.parent / "experiments/representative.py"
    (prior / "experiment.json").write_text(json.dumps({"source_fingerprint": source_fingerprint(),
        "experimental_source_hashes": {str(selector): file_content_hash(selector)}}))
    config = tmp_path / "config.yaml"
    config.write_text("{}")

    def load(data_root, role):
        assert role in {"train", "holdout"}
        if role == "holdout":
            assert (out / "models_locked.json").exists()
        return train if role == "train" else holdout

    def prepare(config, data_root, plan, role, chunk_size, protocol):
        assert role == "holdout"
        assert (out / "models_locked.json").exists()
        assert len(list((out / "jobs").glob("*/manifest.json"))) == 5

    monkeypatch.setattr(study, "load_scaling_artifact", load)
    monkeypatch.setattr(study, "prepare_role", prepare)
    args = Namespace(config=str(config), data_root=str(root), output_dir=str(out),
        prior_choices=str(prior / "locked_choices.json"), budgets=[4], seeds=[0, 1],
        strategies=["random", "quality_representative"], chunk_size=5, declare_only=True)
    study.run(args)
    assert (out / "protocol.json").exists()
    assert not (out / "models_locked.json").exists()
    args.declare_only = False
    study.run(args)
    assert len(pd.read_csv(out / "runs.csv")) == 5
    experiment = json.loads((out / "experiment.json").read_text())
    assert experiment["parameter_search_performed"] is False
    with pytest.raises(ValueError, match="complete"):
        study.run(args)


def test_chunked_preparation_resumes_and_fits_stats_on_complete_candidates(tmp_path, monkeypatch):
    from dataclasses import asdict
    import json
    import pandas as pd
    import soundfile as sf
    from aura.cache.store import file_content_hash
    from aura.config import AuraConfig, CacheConfig, PathsConfig
    from aura.ingest.dataset import SampleRecord
    from experiments.scaling import load_scaling_artifact
    from scripts import prepare_scaling_study as preparation
    from tests.test_encoder import _build_local_encoder

    records = []
    rng = np.random.default_rng(1)
    for i in range(4):
        wave = rng.normal(0, 0.001 * (i + 1), 3200).astype(np.float32)
        wave[800:2400] += 0.2 * np.sin(np.arange(1600) / (10 + i))
        path = tmp_path / f"{i}.wav"
        sf.write(path, wave, 16000)
        record = SampleRecord(f"id{i}", str(path), "a" if i % 2 else "b", f"speaker{i}", i, "train")
        records.append(dict(asdict(record), audio_hash=file_content_hash(path)))
    config = AuraConfig(cache=CacheConfig(path=str(tmp_path / "cache")),
                        paths=PathsConfig(metadata_dir=str(tmp_path / "meta"), embeddings_dir=str(tmp_path / "emb")))
    root = tmp_path / "scaling"
    root.mkdir()
    plan = {"train": records, "holdout": []}
    (root / "partition.json").write_text(json.dumps(plan))
    encoder = _build_local_encoder()
    monkeypatch.setattr(preparation, "Wav2Vec2Encoder", lambda **kwargs: encoder)
    actual_process = preparation.process_split
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(args[0][0].sample_id)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return actual_process(*args, **kwargs)

    monkeypatch.setattr(preparation, "process_split", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        preparation.prepare_role(config, root, plan, "train", 2, None)
    first_manifest = root / "chunks/train/000000/manifest.json"
    saved_hash = file_content_hash(first_manifest)
    monkeypatch.setattr(preparation, "process_split", actual_process)
    preparation.prepare_role(config, root, plan, "train", 2, None)
    assert file_content_hash(first_manifest) == saved_hash
    frame, matrix, _ = load_scaling_artifact(root, "train")
    assert frame.id.tolist() == [r["sample_id"] for r in records]
    assert matrix.shape == (4, 32)
    stats = json.loads((root / "metadata/quality_norm_stats.json").read_text())["stats"]
    assert stats["snr_min"] == frame.snr.min()
    assert stats["snr_max"] == frame.snr.max()
    with pytest.raises(ValueError, match="protocol"):
        preparation.prepare_role(config, root, plan, "holdout", 2, None)


def test_parallel_worker_writes_frozen_benchmark_compatible_checkpoint(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from aura.cache.store import config_fingerprint, file_content_hash
    from scripts import precompute_scaling_jobs as worker
    from scripts.benchmark_scaling_study import make_selector

    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 4)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    frame = pd.DataFrame({"id": [f"id{i}" for i in range(20)], "label": ["a", "b"] * 10,
                          "split": "train", "speaker_id": [f"s{i}" for i in range(20)],
                          "quality_score": np.linspace(0, 1, 20)})
    embeddings = dict(zip(frame.id, matrix))
    monkeypatch.setattr(worker, "STATE", {"frame": frame, "matrix": matrix, "embeddings": embeddings,
        "rows": dict(zip(frame.id, range(20))), "output": tmp_path, "protocol_hash": "frozen",
        "train_hash": "training", "workers": 4})
    job = {"strategy": "quality_representative", "budget": 4,
           "setting": "retain=0.75", "C": 1.0, "source_budget": 4}
    expected = make_selector(job["strategy"], job["setting"]).select(frame, embeddings, 4, 3).selected_ids
    worker.fit_job(job, 3)
    directory = tmp_path / "jobs/quality_representative_B4_seed3"
    ids = json.loads((directory / "selection.json").read_text())["selected_ids"]
    assert ids == expected
    record = json.loads((directory / "manifest.json").read_text())
    assert record["identity"] == config_fingerprint({"job": job, "seed": 3, "protocol": "frozen", "train_manifest": "training"})
    assert all(file_content_hash(path) == digest for path, digest in record["artifacts"].items())
