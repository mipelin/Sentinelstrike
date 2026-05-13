"""Tests for TAK metrics."""

from sentinel.tak_bridge.metrics import compute_tak_metrics


def test_counts_message_count():
    msgs = [
        {"category": "vehicle", "type": "a-f-A-M-F-Q", "success": True},
        {"category": "observation", "type": "a-u-G", "success": True},
    ]
    m = compute_tak_metrics(msgs)
    assert m["message_count"] == 2


def test_counts_categories():
    msgs = [
        {"category": "vehicle", "type": "a-f-A-M-F-Q", "success": True},
        {"category": "vehicle", "type": "a-f-A-M-F-Q", "success": True},
        {"category": "observation", "type": "a-u-G", "success": True},
    ]
    m = compute_tak_metrics(msgs)
    assert m["categories"] == {"vehicle": 2, "observation": 1}


def test_counts_types():
    msgs = [
        {"category": "vehicle", "type": "a-f-A-M-F-Q", "success": True},
        {"category": "waypoint", "type": "b-m-p-w", "success": True},
    ]
    m = compute_tak_metrics(msgs)
    assert m["types"]["a-f-A-M-F-Q"] == 1
    assert m["types"]["b-m-p-w"] == 1


def test_sent_and_failed_counts():
    msgs = [
        {"category": "vehicle", "type": "t1", "success": True},
        {"category": "vehicle", "type": "t1", "success": False},
    ]
    m = compute_tak_metrics(msgs)
    assert m["sent_count"] == 1
    assert m["failed_count"] == 1


def test_empty_messages():
    m = compute_tak_metrics([])
    assert m["message_count"] == 0
    assert m["sent_count"] == 0
