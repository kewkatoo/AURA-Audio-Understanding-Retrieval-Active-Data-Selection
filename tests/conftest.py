from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


def make_tone_clip(
    sample_rate: int = 16000,
    silence_s: float = 0.2,
    tone_s: float = 0.5,
    freq: float = 440.0,
    amplitude: float = 0.6,
) -> np.ndarray:
    """silence -> tone -> silence, a stand-in for a spoken word for tests
    that don't need real speech content, only realistic amplitude/energy
    structure (onset/offset, a non-trivial spectrum)."""
    silence = np.zeros(int(silence_s * sample_rate), dtype=np.float32)
    t = np.linspace(0, tone_s, int(tone_s * sample_rate), endpoint=False)
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.concatenate([silence, tone, silence])


@pytest.fixture
def tmp_wav_file(tmp_path):
    def _make(filename: str = "clip.wav", sample_rate: int = 16000, **kwargs) -> str:
        wav = make_tone_clip(sample_rate=sample_rate, **kwargs)
        path = tmp_path / filename
        sf.write(str(path), wav, sample_rate)
        return str(path)
    return _make


@pytest.fixture
def synthetic_wav_pool(tmp_path):
    """A small pool of synthetic clips with varying amplitude/frequency,
    used for tests that need a nonzero-variance feature distribution
    (e.g. quality normalization)."""
    paths = []
    for i, (amp, freq) in enumerate([(0.1, 300), (0.3, 350), (0.6, 440), (0.9, 500)]):
        wav = make_tone_clip(amplitude=amp, freq=freq)
        path = tmp_path / f"clip_{i}.wav"
        sf.write(str(path), wav, 16000)
        paths.append(str(path))
    return paths
