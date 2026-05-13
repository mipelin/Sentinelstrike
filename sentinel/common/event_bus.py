"""Simple in-process event bus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from loguru import logger

from sentinel.common.events import SystemEvent


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[SystemEvent], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[SystemEvent], None]) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: SystemEvent) -> None:
        for handler in self._handlers.get(event.event_type, []):
            self._safe_call(handler, event)
        for handler in self._handlers.get("*", []):
            self._safe_call(handler, event)

    def clear(self) -> None:
        self._handlers.clear()

    @staticmethod
    def _safe_call(handler: Callable[[SystemEvent], None], event: SystemEvent) -> None:
        try:
            handler(event)
        except Exception:
            logger.exception("EventBus handler error for event_type={}", event.event_type)
