"""FastAPI dashboard application."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .live import LiveRunSession
from .loader import list_run_dirs, load_run_artifacts
from .map_layers import build_map_layers
from .models import RunInfo
from .state import build_operator_queue, build_replay_frames, build_target_cards
from .timeline import build_timeline

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

_INDEX_HTML = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

_LIVE_POLL_INTERVAL = 0.5


def create_app(base_dir: Path = Path("runs")) -> FastAPI:
    app = FastAPI(title="ONS Sentinel Dashboard", version="1.0.0")

    # Active live sessions keyed by run_id
    _sessions: dict[str, LiveRunSession] = {}

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _INDEX_HTML

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/runs", response_model=list[RunInfo])
    async def get_runs():
        return list_run_dirs(base_dir)

    @app.get("/api/runs/latest")
    async def get_latest_run():
        runs = list_run_dirs(base_dir)
        if not runs:
            return {"error": "no runs found"}
        run = runs[0]
        run_path = Path(run.path)
        artifacts = load_run_artifacts(run_path)
        return {
            "run_id": run.run_id,
            "mission_id": run.mission_id,
            "created_at": run.created_at,
            "has_report": run.has_report,
            "has_summary": run.has_summary,
            "summary": _pick_summary(artifacts),
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        artifacts = load_run_artifacts(run_path)
        return {
            "run_id": run_id,
            "artifacts": _artifact_summary(artifacts),
            "summary": _pick_summary(artifacts),
            "targets": [target.model_dump() for target in build_target_cards(artifacts)],
            "operator_queue": [item.model_dump() for item in build_operator_queue(artifacts)],
        }

    @app.get("/api/runs/{run_id}/timeline")
    async def get_timeline(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        artifacts = load_run_artifacts(run_path)
        timeline = build_timeline(artifacts)
        return {"timeline": [e.model_dump() for e in timeline]}

    @app.get("/api/runs/{run_id}/map")
    async def get_map_layers(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        artifacts = load_run_artifacts(run_path)
        layers = build_map_layers(artifacts)
        return layers.model_dump()

    @app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
    async def get_report(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return f"run {run_id} not found"
        report_path = run_path / "report.md"
        if not report_path.exists():
            return "No report available for this run."
        return report_path.read_text(encoding="utf-8")

    @app.get("/api/runs/{run_id}/replay")
    async def get_replay(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        artifacts = load_run_artifacts(run_path)
        replay = build_replay_frames(run_path, artifacts)
        return {"frames": [frame.model_dump() for frame in replay]}

    @app.get("/api/runs/{run_id}/frames/{frame_name}")
    async def get_frame(run_id: str, frame_name: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        frame_path = run_path / frame_name if frame_name == "latest.jpg" else run_path / "frames" / frame_name
        if not frame_path.exists():
            return {"error": f"frame {frame_name} not found"}
        return FileResponse(frame_path)

    @app.get("/video/live/latest")
    async def video_live_latest():
        return StreamingResponse(
            _stream_latest_frames(base_dir),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/video/runs/{run_id}")
    async def video_run(run_id: str):
        run_path = base_dir / run_id
        if not run_path.exists():
            return {"error": f"run {run_id} not found"}
        return StreamingResponse(
            _stream_run_frames(run_path),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws/runs/{run_id}/live")
    async def ws_run_live(websocket: WebSocket, run_id: str):
        await websocket.accept()
        logger.info("WebSocket connected: /ws/runs/{}/live", run_id)
        run_path = base_dir / run_id
        if not run_path.exists():
            await websocket.send_json({"error": f"run {run_id} not found"})
            await websocket.close()
            return

        session = _sessions.get(run_id)
        if session is None:
            session = LiveRunSession(run_id=run_id, run_dir=run_path)
            _sessions[run_id] = session

        try:
            while True:
                payload = session.poll()
                await websocket.send_json(payload)
                await asyncio.sleep(_LIVE_POLL_INTERVAL)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: /ws/runs/{}/live", run_id)

    @app.websocket("/ws/live/latest")
    async def ws_live_latest(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket connected: /ws/live/latest")
        session: LiveRunSession | None = None

        try:
            while True:
                runs = list_run_dirs(base_dir)
                if not runs:
                    await websocket.send_json({"status": "no_runs"})
                    await asyncio.sleep(_LIVE_POLL_INTERVAL)
                    continue

                latest = runs[0]
                latest_id = latest.run_id

                if session is None or session.run_id != latest_id:
                    logger.info("Live session switched to run: {}", latest_id)
                    session = LiveRunSession(
                        run_id=latest_id,
                        run_dir=Path(latest.path),
                    )
                    _sessions[latest_id] = session

                payload = session.poll()
                await websocket.send_json(payload)
                await asyncio.sleep(_LIVE_POLL_INTERVAL)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: /ws/live/latest")

    return app


def _pick_summary(artifacts: dict) -> dict | None:
    for key in ("pipeline_summary", "realtime_metrics"):
        val = artifacts.get(key)
        if val is not None:
            return val
    return None


def _artifact_summary(artifacts: dict) -> dict:
    return {
        "has_metadata": artifacts.get("metadata") is not None,
        "has_mission_plan": artifacts.get("mission_plan") is not None,
        "detections_count": len(artifacts.get("detections", [])),
        "tracks_count": len(artifacts.get("tracks", [])),
        "observations_count": len(artifacts.get("geo_observations", [])),
        "tak_messages_count": len(artifacts.get("tak_messages", [])),
        "operator_decisions_count": len(artifacts.get("operator_decisions", [])),
        "safety_actions_count": len(artifacts.get("safety_actions", [])),
        "events_count": len(artifacts.get("events", [])),
        "frames_count": len(artifacts.get("frames", [])),
        "has_report": artifacts.get("report_md") is not None,
    }


async def _stream_latest_frames(base_dir: Path):
    last_payload: bytes | None = None
    while True:
        runs = list_run_dirs(base_dir)
        if runs:
            frame_path = Path(runs[0].path) / "latest.jpg"
            if frame_path.exists():
                payload = frame_path.read_bytes()
                if payload != last_payload:
                    last_payload = payload
                    yield _mjpeg_chunk(payload)
        await asyncio.sleep(_LIVE_POLL_INTERVAL)


async def _stream_run_frames(run_path: Path):
    frames = sorted((run_path / "frames").glob("frame_*.jpg"))
    if not frames and (run_path / "latest.jpg").exists():
        frames = [run_path / "latest.jpg"]
    while True:
        for frame_path in frames:
            if frame_path.exists():
                yield _mjpeg_chunk(frame_path.read_bytes())
            await asyncio.sleep(0.1)
        if not frames:
            await asyncio.sleep(_LIVE_POLL_INTERVAL)


def _mjpeg_chunk(data: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
