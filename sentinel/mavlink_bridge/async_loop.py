"""Persistent asyncio event loop running in a dedicated daemon thread.

MAVSDK-Python maintains gRPC channels and coroutines bound to the event loop
that created them. Using ``asyncio.run()`` per call destroys the loop and
invalidates those resources, causing ``RuntimeWarning: coroutine was never
awaited`` and ``Event loop is closed`` errors.

This module provides a single long-lived loop so all MAVSDK calls share the
same async context.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any


class AsyncLoopThread:
    """Run a persistent ``asyncio`` event loop in a background daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._closed = False

    # --- internal ---

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # --- public API ---

    def run(self, coro: Coroutine[Any, Any, Any], timeout_s: float | None = None) -> Any:
        """Submit *coro* to the loop and block until it resolves.

        Raises ``RuntimeError`` if the loop has been closed.
        Propagates the coroutine's exception (wrapped in the future result).
        """
        if self._closed:
            raise RuntimeError("AsyncLoopThread is closed — cannot submit coroutines")
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout_s)

    def close(self) -> None:
        """Stop the loop and join the thread.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def closed(self) -> bool:
        return self._closed
