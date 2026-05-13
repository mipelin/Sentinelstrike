"""Mission state builders shared by live and replay dashboard APIs."""

from __future__ import annotations

from pathlib import Path

from .models import ObservationLayer, OperatorQueueItem, ReplayFrame, TargetCard, TrackBox


def build_target_cards(artifacts: dict) -> list[TargetCard]:
    decision_by_track = _latest_decisions_by_track(artifacts.get("operator_decisions", []))
    obs_by_track = _latest_observations_by_track(artifacts.get("geo_observations", []))

    latest_track_by_id: dict[str, dict] = {}
    for trk in artifacts.get("tracks", []):
        track_id = trk.get("track_id")
        if not track_id:
            continue
        latest_track_by_id[track_id] = trk

    cards: list[TargetCard] = []
    for track_id, trk in latest_track_by_id.items():
        meta = trk.get("metadata", {}) if isinstance(trk.get("metadata"), dict) else {}
        obs = obs_by_track.get(track_id)
        decision = decision_by_track.get(track_id)
        status = classify_target_status(trk, decision)
        bbox = _bbox_from_track(trk, meta)
        priority_level = str(meta.get("priority_level", "LOW")).upper()
        suppressed = bool(meta.get("suppressed", False))
        cards.append(
            TargetCard(
                track_id=track_id,
                class_name=trk.get("class_name", ""),
                confidence=float(trk.get("confidence", 0.0) or 0.0),
                age_frames=int(trk.get("age_frames", 0) or 0),
                lost_frames=int(trk.get("lost_frames", 0) or 0),
                status=status,
                confirmation_state=_confirmation_state(decision),
                operator_disposition=_operator_disposition(decision),
                last_seen_utc=trk.get("last_seen_utc", ""),
                first_seen_utc=trk.get("first_seen_utc", ""),
                stale=status in {"STALE", "LOST"},
                reacquired_count=int(meta.get("reacquired_count", 0) or 0),
                frame_id=int(meta.get("frame_id", 0) or 0),
                bbox=bbox,
                history_px=_coerce_history(meta.get("history_px")),
                observation_id=obs.get("observation_id") if obs else None,
                observation_lat=_obs_lat(obs),
                observation_lon=_obs_lon(obs),
                movement_state=str(meta.get("movement_state", "STATIONARY")).upper(),
                average_speed_px_s=float(meta.get("average_speed_px_s", 0.0) or 0.0),
                displacement_px=float(meta.get("displacement_px", 0.0) or 0.0),
                stationary_frames=int(meta.get("stationary_frames", 0) or 0),
                moving_frames=int(meta.get("moving_frames", 0) or 0),
                priority_score=float(meta.get("priority_score", 0.0) or 0.0),
                priority_level=priority_level,
                suppressed=suppressed,
            )
        )

    cards.sort(key=lambda c: (-_priority_sort_key(c.priority_level), -c.priority_score, c.track_id))
    return cards


_PRIORITY_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _priority_sort_key(level: str) -> int:
    return _PRIORITY_ORDER.get(level, 0)


def build_operator_queue(artifacts: dict) -> list[OperatorQueueItem]:
    pending: dict[str, OperatorQueueItem] = {}

    for event in artifacts.get("events", []):
        if event.get("event_type") != "operator_decision_required":
            continue
        payload = event.get("payload", {})
        observation_id = payload.get("observation_id", "")
        if not observation_id:
            continue
        pending[observation_id] = OperatorQueueItem(
            observation_id=observation_id,
            track_id=payload.get("track_id", ""),
            requested_at=event.get("timestamp_utc", ""),
            policy_action=payload.get("policy_action", ""),
            decision_state="pending",
            reason=payload.get("reason", ""),
        )

    for decision in artifacts.get("operator_decisions", []):
        observation_id = decision.get("observation_id", "")
        if not observation_id:
            continue
        item = pending.get(observation_id)
        state = str(decision.get("action", "PENDING")).lower()
        if item is None:
            pending[observation_id] = OperatorQueueItem(
                observation_id=observation_id,
                track_id=decision.get("track_id", ""),
                requested_at=decision.get("timestamp_utc", ""),
                policy_action=decision.get("action", ""),
                decision_state=state,
                reason=decision.get("reason", ""),
            )
        else:
            item.decision_state = state

    return sorted(pending.values(), key=lambda item: item.requested_at)


def build_replay_frames(run_dir: Path, artifacts: dict) -> list[ReplayFrame]:
    frame_names = sorted(p.name for p in (run_dir / "frames").glob("frame_*.jpg"))
    if not frame_names:
        return []

    vehicle_by_frame = {int(v.get("frame_id", -1)): v for v in artifacts.get("vehicle_states", [])}
    obs_by_frame: dict[int, list[ObservationLayer]] = {}
    for obs in artifacts.get("geo_observations", []):
        frame_id = _coerce_frame_id(obs.get("metadata", {}))
        if frame_id is None:
            continue
        obs_by_frame.setdefault(frame_id, []).append(
            ObservationLayer(
                lat=_obs_lat(obs) or 0.0,
                lon=_obs_lon(obs) or 0.0,
                track_id=obs.get("track_id", ""),
                class_name=obs.get("class_name", ""),
                confidence=float(obs.get("confidence", 0.0) or 0.0),
                confirmed=False,
                observation_id=obs.get("observation_id", ""),
                frame_id=frame_id,
            )
        )

    tracks_by_frame: dict[int, list[dict]] = {}
    for trk in artifacts.get("tracks", []):
        meta = trk.get("metadata", {})
        frame_id = _coerce_frame_id(meta)
        if frame_id is None:
            continue
        tracks_by_frame.setdefault(frame_id, []).append(trk)

    decision_by_track = _latest_decisions_by_track(artifacts.get("operator_decisions", []))
    replay: list[ReplayFrame] = []
    for frame_name in frame_names:
        frame_id = int(frame_name.split("_")[1].split(".")[0])
        frame_tracks = tracks_by_frame.get(frame_id, [])
        cards: list[TargetCard] = []
        for trk in frame_tracks:
            meta = trk.get("metadata", {}) if isinstance(trk.get("metadata"), dict) else {}
            track_id = trk.get("track_id", "")
            decision = decision_by_track.get(track_id)
            cards.append(
                TargetCard(
                    track_id=track_id,
                    class_name=trk.get("class_name", ""),
                    confidence=float(trk.get("confidence", 0.0) or 0.0),
                    age_frames=int(trk.get("age_frames", 0) or 0),
                    lost_frames=int(trk.get("lost_frames", 0) or 0),
                    status=classify_target_status(trk, decision),
                    confirmation_state=_confirmation_state(decision),
                    operator_disposition=_operator_disposition(decision),
                    last_seen_utc=trk.get("last_seen_utc", ""),
                    first_seen_utc=trk.get("first_seen_utc", ""),
                    stale=trk.get("status") == "lost",
                    reacquired_count=int(meta.get("reacquired_count", 0) or 0),
                    frame_id=frame_id,
                    bbox=_bbox_from_track(trk, meta),
                    history_px=_coerce_history(meta.get("history_px")),
                    movement_state=str(meta.get("movement_state", "STATIONARY")).upper(),
                    average_speed_px_s=float(meta.get("average_speed_px_s", 0.0) or 0.0),
                    displacement_px=float(meta.get("displacement_px", 0.0) or 0.0),
                    stationary_frames=int(meta.get("stationary_frames", 0) or 0),
                    moving_frames=int(meta.get("moving_frames", 0) or 0),
                    priority_score=float(meta.get("priority_score", 0.0) or 0.0),
                    priority_level=str(meta.get("priority_level", "LOW")).upper(),
                    suppressed=bool(meta.get("suppressed", False)),
                )
            )
        replay.append(
            ReplayFrame(
                frame_id=frame_id,
                frame_name=frame_name,
                timestamp_utc=(vehicle_by_frame.get(frame_id) or {}).get("timestamp_utc", ""),
                vehicle_state=vehicle_by_frame.get(frame_id),
                targets=cards,
                observations=obs_by_frame.get(frame_id, []),
            )
        )
    return replay


def classify_target_status(track: dict, decision: dict | None) -> str:
    action = (decision or {}).get("action", "")
    if action == "CONFIRM_OBSERVATION":
        return "CONFIRMED"
    if action == "REJECT_OBSERVATION":
        return "REJECTED"

    meta = track.get("metadata", {}) if isinstance(track.get("metadata"), dict) else {}
    transition = str(meta.get("last_transition", "")).upper()
    if transition == "REACQUIRED":
        return "REACQUIRED"
    if track.get("status") == "lost":
        return "STALE" if int(track.get("lost_frames", 0) or 0) >= 3 else "LOST"
    return "ACTIVE"


def _latest_decisions_by_track(items: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in items:
        track_id = item.get("track_id", "")
        if track_id:
            out[track_id] = item
    return out


def _latest_observations_by_track(items: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in items:
        track_id = item.get("track_id", "")
        if track_id:
            out[track_id] = item
    return out


def _bbox_from_track(track: dict, meta: dict) -> TrackBox | None:
    predicted = meta.get("predicted_bbox")
    if track.get("status") == "lost" and isinstance(predicted, dict):
        return TrackBox(
            x1=int(predicted.get("x1", 0)),
            y1=int(predicted.get("y1", 0)),
            x2=int(predicted.get("x2", 0)),
            y2=int(predicted.get("y2", 0)),
        )
    bbox = track.get("bbox_xyxy")
    if isinstance(bbox, dict):
        return TrackBox(
            x1=int(bbox.get("x1", 0)),
            y1=int(bbox.get("y1", 0)),
            x2=int(bbox.get("x2", 0)),
            y2=int(bbox.get("y2", 0)),
        )
    if isinstance(predicted, dict):
        return TrackBox(
            x1=int(predicted.get("x1", 0)),
            y1=int(predicted.get("y1", 0)),
            x2=int(predicted.get("x2", 0)),
            y2=int(predicted.get("y2", 0)),
        )
    return None


def _confirmation_state(decision: dict | None) -> str:
    if not decision:
        return "UNREVIEWED"
    action = decision.get("action", "")
    if action == "CONFIRM_OBSERVATION":
        return "CONFIRMED"
    if action == "REJECT_OBSERVATION":
        return "REJECTED"
    return "PENDING"


def _operator_disposition(decision: dict | None) -> str:
    if not decision:
        return "PENDING"
    return str(decision.get("action", "PENDING"))


def _coerce_history(history: object) -> list[list[float]]:
    if not isinstance(history, list):
        return []
    out: list[list[float]] = []
    for point in history[-20:]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            out.append([float(point[0]), float(point[1])])
    return out


def _obs_lat(obs: dict | None) -> float | None:
    if not obs:
        return None
    loc = obs.get("estimated_location", {})
    if isinstance(loc, dict):
        lat = loc.get("lat")
        return float(lat) if lat is not None else None
    return None


def _obs_lon(obs: dict | None) -> float | None:
    if not obs:
        return None
    loc = obs.get("estimated_location", {})
    if isinstance(loc, dict):
        lon = loc.get("lon")
        return float(lon) if lon is not None else None
    return None


def _coerce_frame_id(meta: object) -> int | None:
    if not isinstance(meta, dict):
        return None
    value = meta.get("frame_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
