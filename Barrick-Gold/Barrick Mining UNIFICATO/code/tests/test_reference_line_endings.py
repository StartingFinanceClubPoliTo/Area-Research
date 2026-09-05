"""Cross-platform provenance must retain byte hashes and reject real mutations."""

import hashlib

import pytest

from barrick_unified.multimodel_reporting import _verify_declared_reference, sha256


def _digest(value):
    return hashlib.sha256(value).hexdigest().upper()


@pytest.mark.parametrize("ending,mode", [(b"\n", "raw_bytes"), (b"\r\n", "utf8_python_crlf_to_lf")])
def test_python_reference_records_exact_match_mode(tmp_path, ending, mode):
    path = tmp_path / "reference.py"
    path.write_bytes(b"cost = 1.5" + ending)
    raw_hash = sha256(path)
    entry = {"path": path.name, "sha256": raw_hash}
    declared = _digest(b"cost = 1.5\n")
    _verify_declared_reference(path, entry, declared)
    assert entry["sha256"] == raw_hash
    assert entry["declared_sha256"] == declared
    assert entry["declared_hash_match"] == mode
    if mode != "raw_bytes":
        assert entry["lf_sha256"] == declared


@pytest.mark.parametrize("name,raw", [
    ("reference.py", b"cost = 1.6\r\n"),
    ("reference.py", b"cost  = 1.5\r\n"),
    ("reference.py", b"cost = 1.5\r"),
    ("reference.bin", b"cost = 1.5\r\n"),
    ("reference.py", b"\xffcost = 1.5\r\n"),
])
def test_reference_rejects_changes_beyond_python_crlf(tmp_path, name, raw):
    path = tmp_path / name
    path.write_bytes(raw)
    entry = {"path": name, "sha256": sha256(path)}
    with pytest.raises(ValueError, match="declared reference hash mismatch"):
        _verify_declared_reference(path, entry, _digest(b"cost = 1.5\n"))
