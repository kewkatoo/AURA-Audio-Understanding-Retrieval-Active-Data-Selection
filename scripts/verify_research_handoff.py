"""Read-only, standard-library integrity check of the research handoff snapshot."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path, PurePosixPath
import platform


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_path(root, name):
    relative = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid snapshot path: {name}")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Snapshot path escapes project: {name}")
    return target


def verify(root, snapshot, environment=False):
    failures = []
    for name, expected in snapshot["files"].items():
        target = snapshot_path(root, name)
        if not target.is_file():
            failures.append(f"Missing file: {name}")
        elif digest(target) != expected:
            failures.append(f"Changed file: {name}")
    actual_sources = {
        path.relative_to(root).as_posix()
        for folder in snapshot["source_roots"]
        for path in snapshot_path(root, folder).rglob("*.py")
    }
    if actual_sources != set(snapshot["source_files"]):
        failures.append("Python source inventory differs from snapshot")
    if environment:
        if platform.python_version() != snapshot["python"]:
            failures.append(f"Python version differs: expected {snapshot['python']}, found {platform.python_version()}")
        for package, expected in snapshot["packages"].items():
            try:
                actual = version(package)
            except PackageNotFoundError:
                actual = "missing"
            if actual != expected:
                failures.append(f"Package differs: {package}: expected {expected}, found {actual}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--environment", action="store_true", help="Also compare Python and captured package versions")
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / "handoff/research_snapshot.json"
    expected = (root / "handoff/research_snapshot.sha256").read_text().split()[0]
    if digest(path) != expected:
        raise SystemExit("Snapshot checksum differs")
    snapshot = json.loads(path.read_text())
    failures = verify(root, snapshot, args.environment)
    print(json.dumps(dict(all_checks_passed=not failures,files_checked=len(snapshot["files"]),
        environment_checked=args.environment,failures=failures), indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
