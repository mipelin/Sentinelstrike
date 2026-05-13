"""Real-time loop metrics computation."""

from __future__ import annotations


def compute_realtime_metrics(
    frame_count: int,
    detection_count: int,
    track_count: int,
    geo_observation_count: int,
    tak_message_count: int,
    loop_times_ms: list[float],
    telemetry_stale_count: int,
    started_at_utc: str,
    ended_at_utc: str,
    *,
    wall_clock_s: float = 0.0,
    target_fps: float = 0.0,
    dropped_frames: int = 0,
    capture_fps: float = 0.0,
    reconnect_count: int = 0,
    unique_tracks: int = 0,
    suppressed_geo_updates: int = 0,
    suppressed_tak_messages: int = 0,
) -> dict:
    avg_processing_ms = 0.0
    if loop_times_ms:
        avg_processing_ms = sum(loop_times_ms) / len(loop_times_ms)

    achieved_fps = 0.0
    if wall_clock_s > 0 and frame_count > 0:
        achieved_fps = frame_count / wall_clock_s

    avg_frame_period_ms = 0.0
    if frame_count > 1 and wall_clock_s > 0:
        avg_frame_period_ms = (wall_clock_s / frame_count) * 1000.0

    return {
        "frame_count": frame_count,
        "detection_count": detection_count,
        "track_count": track_count,
        "geo_observation_count": geo_observation_count,
        "tak_message_count": tak_message_count,
        "target_fps": round(target_fps, 2),
        "achieved_fps": round(achieved_fps, 2),
        "average_loop_hz": round(achieved_fps, 2),
        "average_frame_processing_ms": round(avg_processing_ms, 2),
        "average_frame_period_ms": round(avg_frame_period_ms, 2),
        "wall_clock_duration_s": round(wall_clock_s, 3),
        "telemetry_stale_count": telemetry_stale_count,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "dropped_frames": dropped_frames,
        "capture_fps": round(capture_fps, 2),
        "reconnect_count": reconnect_count,
        "unique_tracks": unique_tracks,
        "suppressed_geo_updates": suppressed_geo_updates,
        "suppressed_tak_messages": suppressed_tak_messages,
    }
