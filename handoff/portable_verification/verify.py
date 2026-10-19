"""Verify the frozen v7 study from a copied tree without editing manifests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import sys
from unittest.mock import patch


class RelocatedPaths:
    def __init__(self, original_root, root):
        self.original = PureWindowsPath(original_root)
        if not self.original.is_absolute():
            raise ValueError("Original root must be an absolute Windows path")
        self.root = Path(root).resolve()
        self.mapped = set()

    def __call__(self, value):
        text = os.fspath(value)
        windows = PureWindowsPath(text)
        native = Path(text)
        if native.is_absolute() and native.is_relative_to(self.root):
            target = native
        elif windows.is_absolute():
            try:
                relative = windows.relative_to(self.original)
            except ValueError as error:
                raise ValueError(f"Absolute path outside declared original root: {text}") from error
            target = self.root.joinpath(*relative.parts)
        elif native.is_absolute() or windows.drive or windows.root:
            raise ValueError(f"Unmapped absolute or drive-relative path: {text}")
        else:
            target = self.root / native
        target = target.resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Path escapes relocated tree: {text}")
        self.mapped.add(target)
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Copied project root containing baseline source, results and FSDD")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from scripts.verify_research_handoff import digest, verify as verify_snapshot

    snapshot_path = root / "handoff/research_snapshot.json"
    expected = (root / "handoff/research_snapshot.sha256").read_text().split()[0]
    if digest(snapshot_path) != expected:
        raise ValueError("Baseline snapshot checksum differs")
    snapshot = json.loads(snapshot_path.read_text())
    failures = verify_snapshot(root, snapshot, environment=True)
    if failures:
        raise ValueError("Baseline or environment mismatch: " + "; ".join(failures))
    out = root / "results/cost_confirmation_v7"
    protocol = json.loads((out / "protocol.json").read_text())
    candidates = [PureWindowsPath(name).parents[1] for name in protocol["sources"]
                  if PureWindowsPath(name).name == "study_cost_confirmation.py"]
    if len(candidates) != 1:
        raise ValueError("Cannot identify original root from frozen protocol")
    paths = RelocatedPaths(str(candidates[0]), root)
    from scripts import report_cost_confirmation as reporter
    if not Path(reporter.__file__).resolve().is_relative_to(root):
        raise ValueError("Verifier code was imported from outside the relocated tree")

    def mapped_hash(path):
        return digest(paths(path))

    # Translate only filesystem lookups. Frozen JSON bytes and hashes stay intact.
    with patch.object(reporter, "Path", paths), patch.object(reporter, "file_content_hash", mapped_hash):
        _, folds, _, runs, _, jobs = reporter.verify(out)
    if verify_snapshot(root, snapshot):
        raise ValueError("Baseline files changed during verification")
    print(json.dumps(dict(all_checks_passed=True,root=str(root),original_root=str(paths.original),
        baseline_files=len(snapshot["files"]),models=len(jobs),speakers=len(folds),
        primary_models=int(runs.primary.sum()),raw_hashes=3000,mapped_paths=len(paths.mapped),
        scope="Read-only v7 saved-model verification after relocation; no training or manifest rewriting"),indent=2))


if __name__ == "__main__":
    main()
