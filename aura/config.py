"""
Configuration schema for the AURA Phase 1 pipeline.

Loaded from configs/experiment.yaml. Kept intentionally small in Phase 1:
only what's needed for ingest -> DSP features -> quality -> embeddings.
Later phases (selection, experiments, RAG) will extend this file rather
than replace it, so field names here are considered stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class SampleLimitConfig:
    train: Optional[int] = None
    validation: Optional[int] = None
    test: Optional[int] = None


@dataclass
class DatasetConfig:
    name: str = "speech_commands_v0.02"
    classes: list[str] = field(
        default_factory=lambda: [
            "yes", "no", "up", "down", "left", "right",
            "on", "off", "stop", "go",
        ]
    )
    root: str = "./data/raw"
    sample_limit: SampleLimitConfig = field(default_factory=SampleLimitConfig)


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    vad: bool = True
    # energy-based VAD parameters (documented here since they affect
    # reproducibility and are referenced by aura/ingest/audio.py)
    vad_frame_ms: float = 25.0
    vad_hop_ms: float = 10.0
    vad_energy_percentile: float = 50.0  # frames above this percentile of
    # frame-energy are treated as "speech" for speech_ratio / trimming


@dataclass
class EncoderConfig:
    name: str = "wav2vec2"
    checkpoint: str = "facebook/wav2vec2-base"
    pooling: str = "mean"  # only "mean" implemented in Phase 1
    normalize: bool = True  # L2-normalize the pooled embedding
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    batch_size: int = 8


@dataclass
class CacheConfig:
    path: str = "./data/cache"


@dataclass
class PathsConfig:
    metadata_dir: str = "./data/metadata"
    embeddings_dir: str = "./data/embeddings"


@dataclass
class AuraConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "AuraConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        dataset_raw = raw.get("dataset", {}) or {}
        sample_limit_raw = dataset_raw.get("sample_limit", {}) or {}
        dataset = DatasetConfig(
            name=dataset_raw.get("name", DatasetConfig.name),
            classes=dataset_raw.get("classes", DatasetConfig().classes),
            root=dataset_raw.get("root", DatasetConfig.root),
            sample_limit=SampleLimitConfig(
                train=sample_limit_raw.get("train"),
                validation=sample_limit_raw.get("validation"),
                test=sample_limit_raw.get("test"),
            ),
        )

        audio_raw = raw.get("audio", {}) or {}
        audio = AudioConfig(
            sample_rate=audio_raw.get("sample_rate", AudioConfig.sample_rate),
            mono=audio_raw.get("mono", AudioConfig.mono),
            normalize=audio_raw.get("normalize", AudioConfig.normalize),
            vad=audio_raw.get("vad", AudioConfig.vad),
            vad_frame_ms=audio_raw.get("vad_frame_ms", AudioConfig.vad_frame_ms),
            vad_hop_ms=audio_raw.get("vad_hop_ms", AudioConfig.vad_hop_ms),
            vad_energy_percentile=audio_raw.get(
                "vad_energy_percentile", AudioConfig.vad_energy_percentile
            ),
        )

        encoder_raw = raw.get("encoder", {}) or {}
        encoder = EncoderConfig(
            name=encoder_raw.get("name", EncoderConfig.name),
            checkpoint=encoder_raw.get("checkpoint", EncoderConfig.checkpoint),
            pooling=encoder_raw.get("pooling", EncoderConfig.pooling),
            normalize=encoder_raw.get("normalize", EncoderConfig.normalize),
            device=encoder_raw.get("device", EncoderConfig.device),
            batch_size=encoder_raw.get("batch_size", EncoderConfig.batch_size),
        )

        cache_raw = raw.get("cache", {}) or {}
        cache = CacheConfig(path=cache_raw.get("path", CacheConfig.path))

        paths_raw = raw.get("paths", {}) or {}
        paths = PathsConfig(
            metadata_dir=paths_raw.get("metadata_dir", PathsConfig.metadata_dir),
            embeddings_dir=paths_raw.get("embeddings_dir", PathsConfig.embeddings_dir),
        )

        return AuraConfig(
            dataset=dataset, audio=audio, encoder=encoder, cache=cache, paths=paths
        )
