import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("earlier_verification", Path(__file__).with_name("verify.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_compares_original_keys_without_mutating_report(tmp_path):
    path = tmp_path / "report.json"
    original = r"C:\archive\project"
    path.write_text(json.dumps({"artifacts": {original + r"\results\model": "hash"}}))
    before = path.stat().st_mtime_ns
    generated = json.dumps({"artifacts": {str(tmp_path / "results/model"): "hash"}})
    compare = module.preserved_reports(tmp_path, original, [path])
    assert compare(path, generated) == len(generated)
    assert path.stat().st_mtime_ns == before


def test_rejects_changed_content_and_unexpected_writes(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"value":1}')
    compare = module.preserved_reports(tmp_path, r"C:\archive\project", [path])
    with pytest.raises(ValueError, match="differs"):
        compare(path, '{"value":2}')
    with pytest.raises(ValueError, match="Unexpected"):
        compare(tmp_path / "new.json", '{}')
    assert path.read_text() == '{"value":1}'
    assert not (tmp_path / "new.json").exists()
