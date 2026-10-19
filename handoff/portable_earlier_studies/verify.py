"""Read-only relocation verification for the frozen v5 and v6 studies."""
from contextlib import ExitStack
import argparse
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import sys
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1] / "portable_verification/verify.py"
spec = importlib.util.spec_from_file_location("v7_relocated_paths", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
RelocatedPaths = base.RelocatedPaths


def preserved_reports(root, original, allowed):
    """Compare generated reports using original path keys; never write them."""
    root = Path(root).resolve()
    allowed = {Path(path).resolve() for path in allowed}
    original = PureWindowsPath(original)

    def canonical(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                path = PureWindowsPath(key)
                try:
                    relative = path.relative_to(PureWindowsPath(root))
                except ValueError:
                    normalized = key
                else:
                    normalized = str(original.joinpath(*relative.parts))
                if normalized in result:
                    raise ValueError("Report path normalization produced duplicate keys")
                result[normalized] = canonical(item)
            return result
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    def compare(path, text, *args, **kwargs):
        path = path.resolve()
        if path not in allowed:
            raise ValueError(f"Unexpected verification write: {path}")
        if canonical(json.loads(text)) != json.loads(path.read_text()):
            raise ValueError(f"Regenerated report differs: {path}")
        return len(text)

    return compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["audiomnist", "duplicate-control", "cost"])
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from scripts.verify_research_handoff import digest, verify as verify_snapshot
    snapshot_file = root / "handoff/research_snapshot.json"
    if digest(snapshot_file) != (root / "handoff/research_snapshot.sha256").read_text().split()[0]:
        raise ValueError("Baseline snapshot checksum differs")
    snapshot = json.loads(snapshot_file.read_text())
    failures = verify_snapshot(root, snapshot, environment=True)
    if failures:
        raise ValueError("Baseline or environment mismatch: " + "; ".join(failures))
    # Reuse the previously verified mapper without silently changing that supplement.
    previous = BASE.parent / "artifacts.json"
    if digest(previous) != previous.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Mapper supplement checksum differs")
    previous_manifest = json.loads(previous.read_text())
    relative_mapper = BASE.relative_to(BASE.parents[2]).as_posix()
    if digest(BASE) != previous_manifest["files"][relative_mapper]:
        raise ValueError("Mapper source changed")
    name = {"audiomnist": "audiomnist_v5", "duplicate-control": "duplicate_control_v5", "cost": "acquisition_cost_v6"}[args.stage]
    out = root / "results" / name
    protocol = json.loads((out / "protocol.json").read_text())
    runner = "study_acquisition_cost.py" if args.stage == "cost" else "study_independent_dataset.py"
    originals = [PureWindowsPath(path).parents[1] for path in protocol["sources"] if PureWindowsPath(path).name == runner]
    if len(originals) != 1:
        raise ValueError("Cannot identify original project root")
    paths = RelocatedPaths(str(originals[0]), root)
    from experiments import scaling
    if args.stage == "cost":
        from scripts import report_acquisition_cost as verifier
    else:
        from scripts import verify_independent_study as verifier
    for module in [scaling, verifier]:
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("Verifier imports escaped the copied source tree")
    def mapped_hash(path):
        return digest(paths(path))
    print(f"Verifying relocated {args.stage}", flush=True)
    with ExitStack() as stack:
        for module in [scaling, verifier]:
            stack.enter_context(patch.object(module, "Path", paths))
            stack.enter_context(patch.object(module, "file_content_hash", mapped_hash))
        if args.stage == "cost":
            _, _, _, result = verifier.verify(out)
        else:
            argv = [verifier.__file__] + (["--control"] if args.stage == "duplicate-control" else [])
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(Path, "write_text", preserved_reports(root, paths.original,
                [out / "verification.json", out / "experiment.json"])))
            verifier.main()
            result = json.loads((out / "verification.json").read_text())
    if verify_snapshot(root, snapshot):
        raise ValueError("Copied baseline changed during verification")
    print(json.dumps(dict(all_checks_passed=True,stage=args.stage,models=result["models"],
        raw_hashes_checked=result["raw_hashes_checked"],mapped_paths=len(paths.mapped),baseline_files=len(snapshot["files"]),
        scope="Relocated original semantic verifier: hashes, cohorts, selections, saved metrics and cost/numerical audits where applicable; no model fitting or report writes."),indent=2))


if __name__ == "__main__":
    main()
