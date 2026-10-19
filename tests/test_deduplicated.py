import pandas as pd
import pytest
from experiments.deduplicated import DeduplicatedRandomSelector


def test_duplicate_control_samples_content_once_and_preserves_class_budget():
    meta = pd.DataFrame(dict(id=list("abcdef"), label=["0"] * 3 + ["1"] * 3,
                             split="train", audio_hash=["a", "a", "b", "c", "c", "d"]))
    selector = DeduplicatedRandomSelector()
    result = selector.select(meta, {}, 4, 0)
    chosen = meta.set_index("id").loc[result.selected_ids]
    assert chosen.audio_hash.nunique() == 4
    assert result.class_counts == {"0": 2, "1": 2}
    assert selector.select(meta.sample(frac=1), {}, 4, 0).selected_ids == result.selected_ids
    with pytest.raises(ValueError, match="unique-content"):
        selector.select(meta, {}, 5, 0)
    meta.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="training"):
        selector.select(meta, {}, 4, 0)


def test_conflicting_duplicate_labels_fail():
    meta = pd.DataFrame(dict(id=["a", "b"], label=["0", "1"], split="train", audio_hash="same"))
    with pytest.raises(ValueError, match="conflicting"):
        DeduplicatedRandomSelector().select(meta, {}, 1, 0)
