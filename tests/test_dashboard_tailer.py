"""Tests for JSONL file tailer."""


from sentinel.dashboard.tailer import JsonlTailer


def test_poll_reads_new_lines(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
    tailer = JsonlTailer(p)
    items = tailer.poll()
    assert len(items) == 2
    assert items[0] == {"a": 1}
    assert items[1] == {"a": 2}
    assert tailer.line_count == 2


def test_poll_incremental(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n', encoding="utf-8")
    tailer = JsonlTailer(p)
    tailer.poll()

    # Append more
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"a":2}\n{"a":3}\n')
    items = tailer.poll()
    assert len(items) == 2
    assert tailer.line_count == 3


def test_poll_no_new_data(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n', encoding="utf-8")
    tailer = JsonlTailer(p)
    tailer.poll()
    items = tailer.poll()
    assert items == []


def test_poll_missing_file(tmp_path):
    tailer = JsonlTailer(tmp_path / "missing.jsonl")
    assert tailer.poll() == []


def test_poll_partial_line(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n{"a":2', encoding="utf-8")  # partial last line
    tailer = JsonlTailer(p)
    items = tailer.poll()
    assert len(items) == 1  # only the complete line
    assert tailer.line_count == 1

    # Complete the line
    with open(p, "a", encoding="utf-8") as f:
        f.write("}\n")
    items = tailer.poll()
    assert len(items) == 1
    assert items[0] == {"a": 2}
    assert tailer.line_count == 2


def test_poll_malformed_line(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\nbad line\n{"a":3}\n', encoding="utf-8")
    tailer = JsonlTailer(p)
    items = tailer.poll()
    assert len(items) == 2  # skips malformed
    assert tailer.line_count == 2


def test_reset(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a":1}\n', encoding="utf-8")
    tailer = JsonlTailer(p)
    tailer.poll()
    assert tailer.line_count == 1
    tailer.reset()
    assert tailer.offset == 0
    assert tailer.line_count == 0
    items = tailer.poll()
    assert len(items) == 1  # re-reads from start


def test_properties(tmp_path):
    tailer = JsonlTailer(tmp_path / "x.jsonl")
    assert tailer.path == tmp_path / "x.jsonl"
    assert tailer.offset == 0
    assert tailer.line_count == 0
