"""Cursor-on-Target XML builder.

CoT types used (approximate, v1):
  a-f-A-M-F-Q   — UAV (fixed-wing, military, friendly)
  a-u-G          — Ground unit / observed generic object
  b-m-p-w        — Waypoint
  b-a-o-tbl      — Alert

These are simplified and not doctrinally precise. Future versions should
use proper 2525C/MIL-STD type hierarchies.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta


def cot_time_now(stale_after_s: int = 120) -> tuple[str, str, str]:
    now = datetime.now(UTC)
    start = now
    stale = now + timedelta(seconds=stale_after_s)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return now.strftime(fmt), start.strftime(fmt), stale.strftime(fmt)


def build_cot_event(
    uid: str,
    type_: str,
    lat: float,
    lon: float,
    hae_m: float = 0.0,
    ce_m: float = 9999999.0,
    le_m: float = 9999999.0,
    callsign: str | None = None,
    remarks: str | None = None,
    detail_extra: dict | None = None,
    stale_after_s: int = 120,
) -> str:
    time_str, start_str, stale_str = cot_time_now(stale_after_s)

    event = ET.Element("event")
    event.set("version", "2.0")
    event.set("uid", uid)
    event.set("type", type_)
    event.set("how", "m-g")
    event.set("time", time_str)
    event.set("start", start_str)
    event.set("stale", stale_str)

    point = ET.SubElement(event, "point")
    point.set("lat", f"{lat}")
    point.set("lon", f"{lon}")
    point.set("hae", f"{hae_m}")
    point.set("ce", f"{ce_m}")
    point.set("le", f"{le_m}")

    if callsign is not None or remarks is not None or detail_extra is not None:
        detail = ET.SubElement(event, "detail")

        if callsign is not None:
            contact = ET.SubElement(detail, "contact")
            contact.set("callsign", callsign)

        if remarks is not None:
            remarks_el = ET.SubElement(detail, "remarks")
            remarks_el.text = remarks

        if detail_extra:
            for tag, attrs in detail_extra.items():
                el = ET.SubElement(detail, tag)
                if isinstance(attrs, dict):
                    for k, v in attrs.items():
                        el.set(k, str(v))
                else:
                    el.text = str(attrs)

    return ET.tostring(event, encoding="unicode", xml_declaration=False)


def validate_cot_xml(xml: str) -> bool:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    if root.tag != "event":
        return False
    if root.get("uid") is None:
        return False
    if root.get("type") is None:
        return False
    if root.find("point") is None:
        return False
    return True
