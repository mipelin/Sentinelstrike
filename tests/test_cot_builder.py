"""Tests for CoT XML builder."""

import xml.etree.ElementTree as ET

from sentinel.tak_bridge.cot import build_cot_event, cot_time_now, validate_cot_xml


def test_build_cot_event_parseable():
    xml = build_cot_event(uid="test-1", type_="a-u-G", lat=38.0, lon=-8.0)
    root = ET.fromstring(xml)
    assert root.tag == "event"
    assert root.get("uid") == "test-1"
    assert root.get("type") == "a-u-G"


def test_xml_contains_event_and_point():
    xml = build_cot_event(uid="t", type_="a-u-G", lat=1.0, lon=2.0)
    root = ET.fromstring(xml)
    point = root.find("point")
    assert point is not None
    assert point.get("lat") == "1.0"
    assert point.get("lon") == "2.0"


def test_callsign_appears_when_passed():
    xml = build_cot_event(uid="t", type_="a-u-G", lat=0, lon=0, callsign="ALPHA-1")
    root = ET.fromstring(xml)
    contact = root.find("detail/contact")
    assert contact is not None
    assert contact.get("callsign") == "ALPHA-1"


def test_remarks_appears_when_passed():
    xml = build_cot_event(uid="t", type_="a-u-G", lat=0, lon=0, remarks="test note")
    root = ET.fromstring(xml)
    remarks = root.find("detail/remarks")
    assert remarks is not None
    assert remarks.text == "test note"


def test_validate_cot_xml_true_for_valid():
    xml = build_cot_event(uid="t", type_="a-u-G", lat=0, lon=0)
    assert validate_cot_xml(xml) is True


def test_validate_cot_xml_false_for_invalid():
    assert validate_cot_xml("<notcot/>") is False
    assert validate_cot_xml("not xml at all") is False
    assert validate_cot_xml('<event uid="x"><point lat="0" lon="0"/></event>') is False
    assert validate_cot_xml('<event uid="x" type="a-u-G"><point lat="0" lon="0"/></event>') is True


def test_cot_time_now_returns_three_strings():
    t, s, st = cot_time_now(120)
    assert isinstance(t, str)
    assert isinstance(s, str)
    assert isinstance(st, str)
    assert t == s  # start == time
    assert st != t  # stale is later
