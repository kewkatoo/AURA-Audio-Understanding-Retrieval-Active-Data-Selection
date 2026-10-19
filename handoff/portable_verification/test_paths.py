import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("portable_verify", Path(__file__).with_name("verify.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_remaps_windows_paths_and_relative_paths(tmp_path):
    mapper = module.RelocatedPaths(r"C:\old\project", tmp_path)
    expected = tmp_path / "results" / "model.json"
    assert mapper(r"c:\OLD\project\results\model.json") == expected
    assert mapper("results/model.json") == expected
    assert mapper(expected) == expected


@pytest.mark.parametrize("name", [r"C:\old\project-other\file", r"D:\unrelated\file", "../escape", r"C:relative", r"\rooted", r"C:\old\project\..\escape"])
def test_refuses_unmapped_paths_and_traversal(tmp_path, name):
    mapper = module.RelocatedPaths(r"C:\old\project", tmp_path)
    with pytest.raises(ValueError):
        mapper(name)


def test_missing_copy_never_falls_back_to_original(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "only_original.txt").write_text("must not be read")
    copied = tmp_path / "copy"
    copied.mkdir()
    mapper = module.RelocatedPaths(str(original), copied)
    target = mapper(str(original / "only_original.txt"))
    assert target == copied / "only_original.txt"
    with pytest.raises(FileNotFoundError):
        target.read_bytes()
