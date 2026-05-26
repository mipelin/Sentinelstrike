"""Factory for creating video sources from configuration."""

from __future__ import annotations

from sentinel.common.event_bus import EventBus
from sentinel.config.schema import VideoSourceConfig

from .managed_source import ManagedVideoSource
from .types import VideoSourceType


def create_video_source(
    config: VideoSourceConfig,
    event_bus: EventBus | None = None,
    mission_id: str = "",
):
    """Create a video source from a VideoSourceConfig."""
    source_type = VideoSourceType(config.source_type)

    if source_type == VideoSourceType.GAZEBO_CAMERA:
        from .gazebo_camera_bridge import GazeboCameraBridge

        if not config.gazebo_camera_topic:
            raise ValueError("gazebo_camera_topic is required when source_type is 'gazebo_camera'")
        return GazeboCameraBridge(
            topic=config.gazebo_camera_topic,
            event_bus=event_bus,
            mission_id=mission_id,
            target_fps=config.target_fps,
        )

    label = config.source_type
    kwargs: dict = {
        "reconnect_enabled": config.reconnect_enabled,
        "reconnect_interval_s": config.reconnect_interval_s,
        "frame_width": config.frame_width,
        "frame_height": config.frame_height,
        "event_bus": event_bus,
        "mission_id": mission_id,
        "target_fps": config.target_fps,
    }

    if source_type == VideoSourceType.FILE:
        label = config.file_path
        kwargs["file_path"] = config.file_path
    elif source_type == VideoSourceType.WEBCAM:
        label = f"webcam:{config.webcam_index}"
        kwargs["webcam_index"] = config.webcam_index
    elif source_type == VideoSourceType.RTSP:
        if not config.rtsp_url:
            raise ValueError("rtsp_url is required when source_type is 'rtsp'")
        label = config.rtsp_url
        kwargs["rtsp_url"] = config.rtsp_url

    return ManagedVideoSource(
        source_type=source_type,
        source_label=label,
        **kwargs,
    )
