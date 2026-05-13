"""Tests for JSONL validation."""

from sentinel.quality.jsonl import validate_jsonl_file, validate_jsonl_files


def test_valid_jsonl(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    result = validate_jsonl_file(p)
    assert result["valid"] is True
    assert result["exists"] is True
    assert result["line_count"] == 2


def test_invalid_jsonl(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    result = validate_jsonl_file(p)
    assert result["valid"] is False
    assert result["error"] is not None
    assert "line 2" in result["error"]


def test_nonexistent_file(tmp_path):
    p = tmp_path / "nope.jsonl"
    result = validate_jsonl_file(p)
    assert result["exists"] is False
    assert result["valid"] is False


def test_validate_jsonl_files(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p1.write_text('{"x":1}\n', encoding="utf-8")
    p2 = tmp_path / "b.jsonl"
    p2.write_text("bad\n", encoding="utf-8")
    results = validate_jsonl_files([p1, p2])
    assert len(results) == 2
    assert results[0]["valid"] is True
    assert results[1]["valid"] is False
