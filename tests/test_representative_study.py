from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest

from scripts import study_representative_selection as study
from scripts.study_quality_signals import corruption_variants
from scripts.analyze_confirmation_uncertainty import speaker_bootstrap


def test_study_locks_before_fresh_confirmation_and_never_opens_test(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)

    def dataset(split, n):
        labels = np.array(["a", "b"] * (n // 2))
        x = rng.normal(0, 0.03, (n, 4)).astype(np.float32)
        x[:, 0] += np.where(labels == "a", 1, -1)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        frame = pd.DataFrame({"id": [f"{split}-{i}" for i in range(n)], "label": labels,
                              "split": "train" if split == "train" else "validation",
                              "speaker_id": [f"{split}-{i}" for i in range(n)],
                              "quality_score": rng.uniform(size=n)})
        return frame, x

    train, x_train = dataset("train", 20)
    val, x_val = dataset("validation", 10)
    confirm, x_confirm = dataset("confirmation", 10)

    def load(config, split):
        assert split in {"train", "validation"}, "Study must never load test"
        frame, matrix = (train, x_train) if split == "train" else (val, x_val)
        return frame, matrix, {"encoder": {"checkpoint": "synthetic"}}

    def prepare(config, train, tuning, output, encoder):
        assert (output / "locked_choices.json").exists()
        assert json.loads((output / "protocol.json").read_text())["previous_test_set"].startswith("not opened")
        return confirm, x_confirm

    monkeypatch.setattr(study, "load_split", load)
    monkeypatch.setattr(study, "prepare_confirmation", prepare)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    args = Namespace(config=str(config), output_dir=str(tmp_path / "results"), budgets=[4],
                     seeds=[0, 1], c_values=[1], alphas=[0.25, 1], retain_fractions=[0.5, 0.9])
    result = study.run_study(args)
    assert len(result) == 7
    assert result.confirmation_accuracy.between(0, 1).all()
    assert json.loads((tmp_path / "results/experiment.json").read_text())["test_data_accessed"] is False
    with pytest.raises(ValueError, match="new output"):
        study.run_study(args)


def test_corruptions_have_expected_measured_signal_effects():
    x = np.sin(np.arange(16000) / 10).astype(np.float32) * 0.2
    variants = dict(corruption_variants(x, np.random.default_rng(0)))
    assert not variants["silence"].any()
    assert np.count_nonzero(variants["retain_prefix_0.25"][4000:]) == 0
    assert np.max(np.abs(variants["clip_gain8"])) == 1
    for target in [0, 10, 20]:
        noise = variants[f"noise_{target}db"] - x
        measured_snr = 10 * np.log10(np.mean(x**2) / np.mean(noise**2))
        assert measured_snr == pytest.approx(target, abs=0.01)


def test_speaker_bootstrap_preserves_paired_zero_and_constant_effects():
    speakers = ["a", "a", "b", "c", "c", "c"]
    assert speaker_bootstrap(np.zeros(6), speakers, 100, 1) == (0, 0)
    low, high = speaker_bootstrap(np.full(6, 0.1), speakers, 100, 1)
    assert low == pytest.approx(10)
    assert high == pytest.approx(10)
    with pytest.raises(ValueError):
        speaker_bootstrap(np.ones(6), ["a"] * 6)
