"""Markdown report generator."""

from __future__ import annotations

from pathlib import Path


def write_markdown_report(summary: dict, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# ONS Sentinel Core — Integrated Mission Report\n")

    # Mission
    lines.append("## Mission\n")
    p = summary.get("planner", {})
    lines.append(f"- Mission ID: `{summary.get('mission_id', '')}`")
    lines.append(f"- Run directory: `{summary.get('run_dir', '')}`")
    lines.append(f"- Planner version: {p.get('version', '')}")
    lines.append(f"- Waypoints: {p.get('waypoint_count', 0)}")
    lines.append(f"- Total distance: {p.get('total_distance_m', 0):.1f} m")
    lines.append(f"- Estimated duration: {p.get('estimated_duration_s', 0):.1f} s")
    lines.append("")

    # Perception
    perc = summary.get("perception", {})
    lines.append("## Perception\n")
    lines.append(f"- Frames processed: {perc.get('frame_count', 0)}")
    lines.append(f"- Detections: {perc.get('detection_count', 0)}")
    lines.append("")

    # Tracking
    trk = summary.get("tracking", {})
    lines.append("## Tracking\n")
    lines.append(f"- Total tracks: {trk.get('total_tracks', 0)}")
    lines.append(f"- Active: {trk.get('active_tracks', 0)}")
    lines.append(f"- Lost: {trk.get('lost_tracks', 0)}")
    lines.append(f"- Terminated: {trk.get('terminated_tracks', 0)}")
    lines.append("")

    # Geolocalization
    geo = summary.get("geolocalization", {})
    lines.append("## Geolocalization\n")
    lines.append(f"- Observations: {geo.get('observation_count', 0)}")
    lines.append("")

    # TAK
    tak = summary.get("tak", {})
    lines.append("## TAK/C2 Output\n")
    lines.append(f"- Messages: {tak.get('message_count', 0)}")
    lines.append(f"- Sent: {tak.get('sent_count', 0)}")
    lines.append(f"- Failed: {tak.get('failed_count', 0)}")
    lines.append("")

    # MAVLink
    mav = summary.get("mavlink", {})
    lines.append("## MAVLink Simulation\n")
    lines.append(f"- Commands executed: {mav.get('command_count', 0)}")
    lines.append(f"- Success: {mav.get('success_count', 0)}")
    lines.append(f"- Failed: {mav.get('failed_count', 0)}")
    lines.append("")

    # Artifacts
    artifacts = summary.get("artifacts", {})
    lines.append("## Artifacts\n")
    for _label, filename in artifacts.items():
        lines.append(f"- `{filename}`")
    lines.append("")

    # Limitations
    lines.append("## Limitations\n")
    lines.append("- Laboratory/simulation pipeline.")
    lines.append("- Approximate flat-ground geolocalization.")
    lines.append("- Mock perception/MAVLink by default.")
    lines.append("- No control of physical UAV hardware in this demo.")
    lines.append("- No bidirectional TAK command processing.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
