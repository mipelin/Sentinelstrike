"""TAK Bridge — orchestrates CoT message generation and delivery."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from loguru import logger

from sentinel.common.event_bus import EventBus
from sentinel.common.events import SystemEvent, make_event
from sentinel.common.time import utc_now_iso
from sentinel.common.types import GeoObservation, MissionPlan, VehicleState
from sentinel.config.schema import TakConfig

from .mapper import (
    geo_observation_to_cot,
    mission_plan_waypoints_to_cot,
    system_event_to_cot,
    vehicle_state_to_cot,
)
from .metrics import compute_tak_metrics
from .transport import DryRunTakTransport, UdpTakTransport


class TakBridge:
    def __init__(
        self,
        config: TakConfig,
        mission_id: str,
        event_bus: EventBus | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._mission_id = mission_id
        self._event_bus = event_bus
        self._run_dir = run_dir
        self._message_records: list[dict] = []
        self._msg_file = None
        self._last_vehicle_publish: float = 0.0
        self._last_track_publish: dict[str, float] = {}
        self._suppressed_count: int = 0
        self._total_sent: int = 0

        transport: DryRunTakTransport | UdpTakTransport
        if config.mode == "dry_run":
            transport = DryRunTakTransport()
        elif config.mode == "udp":
            transport = UdpTakTransport(config.cot_host, config.cot_port)
        else:
            raise ValueError(f"Unsupported TAK mode: {config.mode}")
        self._transport = transport

        if run_dir and config.save_tak_messages_jsonl:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._msg_file = (run_dir / "tak_messages.jsonl").open("w", encoding="utf-8")

    def _publish(self, event_type: str, **payload: object) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                make_event(
                    mission_id=self._mission_id,
                    source="tak_bridge",
                    event_type=event_type,
                    payload=payload,
                )
            )

    def _extract_uid_type(self, xml: str) -> tuple[str, str]:
        try:
            root = ET.fromstring(xml)
            return root.get("uid", ""), root.get("type", "")
        except ET.ParseError:
            return "", ""

    def _record_and_send(self, category: str, xml: str | None) -> bool:
        if xml is None:
            return False
        uid, type_ = self._extract_uid_type(xml)
        try:
            self._transport.send(xml)
            record = {
                "timestamp_utc": utc_now_iso(),
                "mission_id": self._mission_id,
                "category": category,
                "uid": uid,
                "type": type_,
                "success": True,
                "xml": xml,
            }
            self._message_records.append(record)
            if self._msg_file:
                self._msg_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._publish("tak_message_sent", category=category, uid=uid, type=type_)
            return True
        except Exception as exc:
            record = {
                "timestamp_utc": utc_now_iso(),
                "mission_id": self._mission_id,
                "category": category,
                "uid": uid,
                "type": type_,
                "success": False,
                "error": str(exc),
                "xml": xml,
            }
            self._message_records.append(record)
            if self._msg_file:
                self._msg_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._publish("tak_message_failed", category=category, error=str(exc))
            return False

    def send_vehicle_state(self, state: VehicleState) -> bool:
        now = time.monotonic()
        if self._total_sent >= self._config.max_messages_per_run:
            self._suppressed_count += 1
            return False
        if now - self._last_vehicle_publish < self._config.publish_vehicle_every_s:
            self._suppressed_count += 1
            return False
        xml = vehicle_state_to_cot(state, self._config.callsign, self._config.stale_after_s)
        ok = self._record_and_send("vehicle", xml)
        if ok:
            self._last_vehicle_publish = now
            self._total_sent += 1
        return ok

    def send_geo_observation(self, obs: GeoObservation) -> bool:
        now = time.monotonic()
        # Suppress TAK for tentative, and LOW+STATIONARY suppressed tracks
        obs_meta = obs.metadata or {}
        if obs_meta.get("confirmation_state") == "TENTATIVE":
            self._suppressed_count += 1
            return False
        if obs_meta.get("suppressed") and obs_meta.get("priority_level") == "LOW":
            self._suppressed_count += 1
            return False
        if self._total_sent >= self._config.max_messages_per_run:
            self._suppressed_count += 1
            return False
        last = self._last_track_publish.get(obs.track_id, 0.0)
        if now - last < self._config.publish_track_every_s:
            self._suppressed_count += 1
            return False
        xml = geo_observation_to_cot(obs, self._config.callsign, self._config.stale_after_s)
        ok = self._record_and_send("observation", xml)
        if ok:
            self._last_track_publish[obs.track_id] = now
            self._total_sent += 1
        return ok

    def send_mission_plan(self, plan: MissionPlan) -> int:
        messages = mission_plan_waypoints_to_cot(plan, self._config.callsign, self._config.stale_after_s)
        sent = 0
        for xml in messages:
            if self._record_and_send("waypoint", xml):
                sent += 1
        return sent

    def send_system_alert(self, event: SystemEvent, lat: float, lon: float) -> bool:
        xml = system_event_to_cot(event, lat, lon, self._config.callsign, self._config.stale_after_s)
        return self._record_and_send("alert", xml)

    @property
    def suppressed_count(self) -> int:
        return self._suppressed_count

    def close(self) -> dict:
        self._transport.close()
        metrics = compute_tak_metrics(self._message_records)
        metrics["suppressed_tak_messages"] = self._suppressed_count

        if self._run_dir:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            (self._run_dir / "tak_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if self._msg_file:
            self._msg_file.close()

        self._publish("tak_bridge_completed", **metrics)
        logger.info(
            "TAK bridge completed: {} messages, {} suppressed, mode={}",
            metrics["message_count"],
            self._suppressed_count,
            self._config.mode,
        )
        return metrics
