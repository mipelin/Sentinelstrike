"""Tests for realtime rate limiter."""

import time

import pytest

from sentinel.realtime.rate_limiter import RateLimiter


def test_start_initializes_anchor():
    rl = RateLimiter(target_hz=10.0)
    rl.start()
    assert rl.tick_count == 0


def test_wait_increments_tick_count():
    rl = RateLimiter(target_hz=1000.0, sleep_enabled=False)
    rl.start()
    rl.wait()
    assert rl.tick_count == 1
    rl.wait()
    assert rl.tick_count == 2


def test_wait_no_sleep_is_fast():
    rl = RateLimiter(target_hz=5.0, sleep_enabled=False)
    rl.start()
    start = time.monotonic()
    for _ in range(100):
        rl.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


def test_target_hz_property():
    rl = RateLimiter(target_hz=7.5)
    assert rl.target_hz == 7.5


def test_period_s():
    rl = RateLimiter(target_hz=10.0)
    assert abs(rl.period_s - 0.1) < 0.001


def test_zero_hz_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        RateLimiter(target_hz=0.0)


def test_negative_hz_raises():
    with pytest.raises(ValueError, match="must be > 0"):
        RateLimiter(target_hz=-5.0)


def test_sleep_produces_near_target_rate():
    rl = RateLimiter(target_hz=50.0, sleep_enabled=True)
    rl.start()
    n = 5
    start = time.monotonic()
    for _ in range(n):
        rl.wait()
    elapsed = time.monotonic() - start
    achieved = n / elapsed
    # Allow 30% tolerance for OS sleep imprecision
    assert abs(achieved - 50.0) < 20.0


def test_no_drift_over_frames():
    rl = RateLimiter(target_hz=100.0, sleep_enabled=True)
    rl.start()
    n = 10
    start = time.monotonic()
    for _ in range(n):
        rl.wait()
    elapsed = time.monotonic() - start
    expected = n / 100.0
    # Should be within 20ms of expected total duration
    assert abs(elapsed - expected) < 0.05


def test_reset_clears_state():
    rl = RateLimiter(target_hz=10.0, sleep_enabled=False)
    rl.start()
    rl.wait()
    rl.wait()
    assert rl.tick_count == 2
    rl.reset()
    assert rl.tick_count == 0


def test_wait_without_start_auto_initializes():
    rl = RateLimiter(target_hz=1000.0, sleep_enabled=False)
    rl.wait()
    assert rl.tick_count == 1
