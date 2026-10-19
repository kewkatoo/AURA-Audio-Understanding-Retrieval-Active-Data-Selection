import json
import numpy as np
import pandas as pd
import pytest
from scripts import study_independent_dataset as study
from scripts.report_independent_study import paired_comparisons


def test_worker_checkpoint_is_reused_and_tampering_rejected(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    meta = pd.DataFrame(dict(id=[str(i) for i in range(20)], label=["0", "1"] * 10,
                             split="train", audio_hash=[str(i) for i in range(20)]))
    monkeypatch.setattr(study, "_TRAIN", (meta, x, dict(zip(meta.id, x)), dict(zip(meta.id, range(20)))))
    job = dict(strategy="deduplicated_random", budget=10, seed=0, C=1, setting="default")
    payload = job, str(tmp_path), "fixed_protocol"
    folder = study.fit_job(payload)
    assert study.fit_job(payload) == folder
    selection = tmp_path / "jobs/deduplicated_random_B10_seed0/selection.json"
    selection.write_text(json.dumps(["altered"]))
    with pytest.raises(ValueError, match="Changed"):
        study.fit_job(payload)


def test_comparisons_align_ids_and_reject_missing_pairs():
    rows = []
    for strategy in ["random", "representative"]:
        for seed in range(2):
            for i in range(4):
                rows.append(dict(strategy=strategy, budget=100, seed=seed, id=str(i),
                                 speaker_id=str(i // 2), label="0", prediction="0" if strategy == "representative" else "1"))
    frame = pd.DataFrame(rows).sample(frac=1, random_state=0)
    result = paired_comparisons(frame, ["random"])
    assert result.difference_pp.iloc[0] == 100
    assert result.low_pp.iloc[0] == 100
    with pytest.raises(ValueError, match="cohorts"):
        paired_comparisons(frame.iloc[1:], ["random"])


def test_declarations_cannot_be_changed(tmp_path):
    path = tmp_path / "protocol.json"
    study.immutable_json(path, {"seed": 0})
    study.immutable_json(path, {"seed": 0})
    with pytest.raises(ValueError, match="differs"):
        study.immutable_json(path, {"seed": 1})
