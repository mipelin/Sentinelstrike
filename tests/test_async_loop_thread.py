"""Tests for AsyncLoopThread."""

import asyncio
import time

from sentinel.mavlink_bridge.async_loop import AsyncLoopThread


def test_run_simple_coroutine_returns_result():
    loop = AsyncLoopThread()
    result = loop.run(asyncio.sleep(0, result=42))
    assert result == 42
    loop.close()


def test_run_several_coroutines_does_not_close_loop():
    loop = AsyncLoopThread()
    for i in range(5):
        result = loop.run(asyncio.sleep(0, result=i))
        assert result == i
    assert not loop.closed
    loop.close()


def test_close_idempotent():
    loop = AsyncLoopThread()
    loop.close()
    loop.close()  # second call must not raise
    assert loop.closed


def test_run_after_close_raises_runtime_error():
    loop = AsyncLoopThread()
    loop.close()
    try:
        loop.run(asyncio.sleep(0))
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as exc:
        assert "closed" in str(exc).lower()


def test_run_propagates_exception():
    loop = AsyncLoopThread()

    async def _fail():
        raise ValueError("boom")

    try:
        loop.run(_fail())
        raise AssertionError("Should have raised ValueError")
    except ValueError as exc:
        assert "boom" in str(exc)
    finally:
        loop.close()


def test_run_respects_timeout():
    loop = AsyncLoopThread()

    async def _slow():
        await asyncio.sleep(10)

    try:
        loop.run(_slow(), timeout_s=0.2)
        raise AssertionError("Should have timed out")
    except TimeoutError:
        pass
    finally:
        loop.close()


def test_close_stops_promptly():
    loop = AsyncLoopThread()
    t0 = time.monotonic()
    loop.close()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"close took {elapsed:.1f}s — thread join may be stuck"
