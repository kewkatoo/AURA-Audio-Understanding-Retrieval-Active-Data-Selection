"""Fingerprints for invalidating artifacts when code or dependencies change."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

PIPELINE_VERSION = "2"


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in
            ("numpy", "scipy", "librosa", "soundfile", "torch", "torchaudio", "transformers")}


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "aura").rglob("*.py")) + [root / "scripts/prepare_data.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
