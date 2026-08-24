"""
Google Speech Commands v0.02 loading via torchaudio, filtered to the
configured class subset, using the dataset's OWN official
training/validation/testing split (validation_list.txt / testing_list.txt).

We deliberately do not implement any custom splitting logic. torchaudio's
`SPEECHCOMMANDS(subset=...)` already encodes the official split; reproducing
it by hand would be an easy way to silently introduce train/test leakage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torchaudio.datasets import SPEECHCOMMANDS


@dataclass
class SampleRecord:
    """One dataset sample, resolved to a concrete file on disk. This is the
    unit passed downstream to preprocessing / feature extraction / encoding.
    """
    sample_id: str          # deterministic id: f"{label}_{speaker_id}_{utterance_number}"
    audio_path: str
    label: str
    speaker_id: Optional[str]
    utterance_number: Optional[int]
    split: str               # "train" | "validation" | "test"


class FilteredSpeechCommands:
    """Wraps torchaudio.datasets.SPEECHCOMMANDS, restricts to a class
    allowlist, and exposes plain file-path records rather than decoded
    waveforms -- decoding happens later in aura/ingest/audio.py so this
    class stays cheap to construct and iterate for metadata purposes.
    """

    _SPLIT_TO_TORCHAUDIO_SUBSET = {
        "train": "training",
        "validation": "validation",
        "test": "testing",
    }

    def __init__(
        self,
        root: str,
        classes: list[str],
        split: str,
        download: bool = True,
    ):
        if split not in self._SPLIT_TO_TORCHAUDIO_SUBSET:
            raise ValueError(
                f"split must be one of {list(self._SPLIT_TO_TORCHAUDIO_SUBSET)}, got {split!r}"
            )
        self.root = root
        self.classes = set(classes)
        self.split = split

        Path(root).mkdir(parents=True, exist_ok=True)
        subset = self._SPLIT_TO_TORCHAUDIO_SUBSET[split]
        self._dataset = SPEECHCOMMANDS(
            root=root, download=download, subset=subset
        )

        # torchaudio stores each item's file path internally; we access it
        # via the dataset's _walker (list of file paths) which is the
        # public-ish mechanism used across torchaudio versions for this
        # dataset. We filter to our class allowlist up front so downstream
        # code never sees excluded classes.
        self._records: list[SampleRecord] = []
        for path in self._dataset._walker:
            relpath = os.path.relpath(path, self._dataset._path)
            label = os.path.dirname(relpath)
            if label not in self.classes:
                continue
            filename = os.path.basename(relpath)
            stem, _ = os.path.splitext(filename)
            try:
                speaker_id, utterance_str = stem.split("_nohash_")
                utterance_number = int(utterance_str)
            except ValueError:
                speaker_id, utterance_number = stem, None
            sample_id = f"{label}_{speaker_id}_{utterance_number}"
            self._records.append(
                SampleRecord(
                    sample_id=sample_id,
                    audio_path=path,
                    label=label,
                    speaker_id=speaker_id,
                    utterance_number=utterance_number,
                    split=split,
                )
            )

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def records(
        self, sample_limit: Optional[int] = None, seed: int = 0
    ) -> list[SampleRecord]:
        """Returns records, optionally capped to `sample_limit`.

        Capping is a *stratified* deterministic sample (roughly equal
        count per class, shuffled with a fixed seed) rather than a raw
        head-of-list truncation, so small dev runs still cover all
        classes. Set sample_limit=None to use the full split.
        """
        if sample_limit is None:
            return list(self._records)

        import random

        by_class: dict[str, list[SampleRecord]] = {}
        for r in self._records:
            by_class.setdefault(r.label, []).append(r)

        rng = random.Random(seed)
        for label in by_class:
            rng.shuffle(by_class[label])

        n_classes = len(by_class)
        per_class = max(1, sample_limit // n_classes)

        selected: list[SampleRecord] = []
        for label, recs in by_class.items():
            selected.extend(recs[:per_class])

        # Trim any overshoot from rounding, deterministically.
        rng.shuffle(selected)
        return selected[:sample_limit]
