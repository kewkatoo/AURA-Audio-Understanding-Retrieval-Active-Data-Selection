"""
AudioEncoder interface.

Phase 1 implements exactly one concrete encoder (Wav2Vec2, see
aura/encoders/wav2vec2.py). This interface exists so a future encoder
(HuBERT, BEATs, CLAP, ...) can be swapped in later without touching
scoring/selection/experiment code, which only ever depends on this
interface, not on any encoder-specific class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EncoderInfo:
    name: str
    checkpoint: str
    embedding_dim: int      # recorded automatically from the loaded model,
                             # never hard-coded
    pooling: str
    l2_normalized: bool
    device: str


class AudioEncoder(ABC):
    """All encoders take a batch of mono float32 waveforms at a fixed
    sample rate and return a batch of fixed-size embedding vectors."""

    @property
    @abstractmethod
    def info(self) -> EncoderInfo:
        ...

    @abstractmethod
    def encode_batch(
        self, waveforms: list[np.ndarray], sample_rate: int
    ) -> np.ndarray:
        """Returns an array of shape (batch_size, embedding_dim), float32."""
        ...

    def encode_one(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.encode_batch([waveform], sample_rate)[0]
