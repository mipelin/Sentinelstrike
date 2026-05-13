"""Tests for TAK transport."""

import pytest

from sentinel.tak_bridge.cot import build_cot_event
from sentinel.tak_bridge.transport import DryRunTakTransport, UdpTakTransport


def test_dry_run_stores_messages():
    t = DryRunTakTransport()
    xml = build_cot_event(uid="t1", type_="a-u-G", lat=0, lon=0)
    t.send(xml)
    assert len(t.sent_messages) == 1
    assert t.sent_messages[0] == xml


def test_dry_run_rejects_invalid_xml():
    t = DryRunTakTransport()
    with pytest.raises(ValueError):
        t.send("<invalid")


def test_udp_transport_instantiable():
    t = UdpTakTransport("127.0.0.1", 8087)
    t.close()


def test_dry_run_close_noop():
    t = DryRunTakTransport()
    t.close()  # should not raise
