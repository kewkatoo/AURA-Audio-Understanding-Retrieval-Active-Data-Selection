"""Run the legacy independent-study verifier without refreshing its reports."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.verify_independent_study import main as legacy_main


def preserved_writer(allowed):
    allowed = {path.resolve() for path in allowed}

    def compare(path, text, *args, **kwargs):
        if path.resolve() not in allowed:
            raise ValueError(f"Unexpected verification write: {path}")
        if not path.is_file() or path.read_text() != text:
            raise ValueError(f"Regenerated report differs from saved evidence: {path}")
        return len(text)

    return compare


def main():
    root = Path("results/duplicate_control_v5" if "--control" in sys.argv else "results/audiomnist_v5")
    with patch.object(Path, "write_text", preserved_writer([root / "verification.json", root / "experiment.json"])):
        legacy_main()


if __name__ == "__main__":
    main()
