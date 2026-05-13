"""TAK transport layer — dry-run and UDP."""

from __future__ import annotations

import socket
from typing import Protocol, runtime_checkable

from .cot import validate_cot_xml


@runtime_checkable
class TakTransport(Protocol):
    def send(self, xml: str) -> None: ...
    def close(self) -> None: ...


class DryRunTakTransport:
    """Stores messages in memory. No network activity."""

    def __init__(self) -> None:
        self.sent_messages: list[str] = []

    def send(self, xml: str) -> None:
        if not validate_cot_xml(xml):
            raise ValueError("Invalid CoT XML")
        self.sent_messages.append(xml)

    def close(self) -> None:
        pass


class UdpTakTransport:
    """Sends CoT XML as UTF-8 UDP datagrams."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, xml: str) -> None:
        data = xml.encode("utf-8")
        self._sock.sendto(data, (self._host, self._port))

    def close(self) -> None:
        self._sock.close()
