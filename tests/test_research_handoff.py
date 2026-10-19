from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from scripts import verify_research_handoff as handoff
from scripts.verify_saved_independent_study import preserved_writer


def test_integrity_detects_changed_missing_and_added_source(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    code = source / "main.py"
    code.write_text("original")
    artifact = tmp_path / "result.csv"
    artifact.write_text("saved")
    snapshot = dict(files={"src/main.py": handoff.digest(code), "result.csv": handoff.digest(artifact)},
        source_roots=["src"], source_files=["src/main.py"])
    assert handoff.verify(tmp_path, snapshot) == []
    code.write_text("changed")
    artifact.unlink()
    (source / "extra.py").write_text("added")
    failures = handoff.verify(tmp_path, snapshot)
    assert "Changed file: src/main.py" in failures
    assert "Missing file: result.csv" in failures
    assert "Python source inventory differs from snapshot" in failures


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "src\\outside"])
def test_snapshot_paths_cannot_escape_project(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid snapshot path"):
        handoff.snapshot_path(tmp_path, name)


def test_environment_check_reports_missing_packages_and_python(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff.platform, "python_version", lambda: "3.11.0")
    def missing(package):
        raise PackageNotFoundError(package)
    monkeypatch.setattr(handoff, "version", missing)
    snapshot = dict(files={}, source_roots=[], source_files=[], python="3.11.1", packages={"absent": "1.0"})
    failures = handoff.verify(tmp_path, snapshot, environment=True)
    assert len(failures) == 2
    assert "Python version differs" in failures[0]
    assert "found missing" in failures[1]


def test_legacy_verifier_wrapper_compares_without_writing(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("saved")
    before = report.stat().st_mtime_ns
    compare = preserved_writer([report])
    assert compare(report, "saved") == 5
    assert report.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="differs"):
        compare(report, "changed")
    with pytest.raises(ValueError, match="Unexpected"):
        compare(tmp_path / "unexpected", "new")
    assert report.read_text() == "saved"
    assert not (tmp_path / "unexpected").exists()
