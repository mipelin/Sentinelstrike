"""Tests for realtime clock helpers."""

import time

from sentinel.realtime.clock import elapsed_s, monotonic_now_s, utc_now_iso


def test_monotonic_now_s_increases():
    a = monotonic_now_s()
    time.sleep(0.01)
    b = monotonic_now_s()
    assert b > a


def test_utc_now_iso_format():
    ts = utc_now_iso()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts) == 27  # YYYY-MM-DDTHH:MM:SS.ffffffZ


def test_elapsed_s():
    start = monotonic_now_s()
    time.sleep(0.02)
    e = elapsed_s(start)
    assert e >= 0.01
    assert e < 1.0
