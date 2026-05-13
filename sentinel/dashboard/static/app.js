const API = '';
const DEFAULT_VIDEO_WIDTH = 640;
const DEFAULT_VIDEO_HEIGHT = 480;

let map = null;
let layerGroups = {};
let liveWs = null;
let isLive = false;
let currentRunId = null;
let selectedTrackId = null;
let replayFrames = [];
let replayIndex = 0;
let replayTimer = null;
let replaySpeed = 1.0;
let currentTimeline = [];
let currentTargets = [];
let videoFrameSize = { width: DEFAULT_VIDEO_WIDTH, height: DEFAULT_VIDEO_HEIGHT };
let latestMapData = null;
let latestTelemetry = null;

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    bindEvents();
    loadRuns();
});

function bindEvents() {
    document.getElementById('btn-load').addEventListener('click', loadSelectedRun);
    document.getElementById('btn-latest').addEventListener('click', loadLatestRun);
    document.getElementById('btn-live').addEventListener('click', startLiveLatest);
    document.getElementById('btn-stop-live').addEventListener('click', stopLive);
    document.getElementById('btn-play').addEventListener('click', startReplay);
    document.getElementById('btn-pause').addEventListener('click', pauseReplay);
    document.getElementById('speed-select').addEventListener('change', (event) => {
        replaySpeed = Number(event.target.value || 1);
        if (replayTimer) {
            startReplay();
        }
    });
    document.getElementById('scrubber').addEventListener('input', (event) => {
        pauseReplay();
        replayIndex = Number(event.target.value || 0);
        renderReplayFrame(replayIndex);
    });
    document.getElementById('video-feed').addEventListener('load', (event) => {
        videoFrameSize = {
            width: event.target.naturalWidth || DEFAULT_VIDEO_WIDTH,
            height: event.target.naturalHeight || DEFAULT_VIDEO_HEIGHT,
        };
        renderVideoFocus();
    });
}

function initMap() {
    map = L.map('map', { zoomControl: false }).setView([38.0, -8.0], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19,
    }).addTo(map);
    layerGroups = {
        waypoints: L.layerGroup().addTo(map),
        observations: L.layerGroup().addTo(map),
        uav: L.layerGroup().addTo(map),
        track: L.layerGroup().addTo(map),
        highlight: L.layerGroup().addTo(map),
    };
}

function setMode(mode) {
    const badge = document.getElementById('mode-badge');
    if (mode === 'live') {
        isLive = true;
        badge.textContent = 'Live';
        badge.className = 'mode-badge mode-live';
        document.getElementById('btn-live').style.display = 'none';
        document.getElementById('btn-stop-live').style.display = '';
    } else {
        isLive = false;
        badge.textContent = 'Replay';
        badge.className = 'mode-badge mode-replay';
        document.getElementById('btn-live').style.display = '';
        document.getElementById('btn-stop-live').style.display = 'none';
    }
}

function setStatus(message) {
    document.getElementById('status-bar').textContent = message;
}

async function loadRuns() {
    const response = await fetch(`${API}/api/runs`);
    const runs = await response.json();
    const select = document.getElementById('run-select');
    select.innerHTML = '';
    if (!runs.length) {
        select.innerHTML = '<option value="">No runs</option>';
        return;
    }
    runs.forEach((run) => {
        const option = document.createElement('option');
        option.value = run.run_id;
        option.textContent = `${run.run_id} (${run.mission_id || 'mission'})`;
        select.appendChild(option);
    });
}

function loadSelectedRun() {
    const runId = document.getElementById('run-select').value;
    if (runId) {
        stopLive();
        loadRun(runId);
    }
}

function loadLatestRun() {
    const select = document.getElementById('run-select');
    if (select.options.length > 0) {
        select.selectedIndex = 0;
        loadSelectedRun();
    }
}

async function loadRun(runId) {
    currentRunId = runId;
    selectedTrackId = null;
    setMode('replay');
    setStatus(`Loading ${runId}...`);

    const [runResp, timelineResp, mapResp, replayResp] = await Promise.all([
        fetch(`${API}/api/runs/${runId}`),
        fetch(`${API}/api/runs/${runId}/timeline`),
        fetch(`${API}/api/runs/${runId}/map`),
        fetch(`${API}/api/runs/${runId}/replay`),
    ]);

    const runData = await runResp.json();
    const timelineData = await timelineResp.json();
    const mapData = await mapResp.json();
    const replayData = await replayResp.json();

    currentTimeline = timelineData.timeline || [];
    replayFrames = replayData.frames || [];
    replayIndex = 0;
    latestMapData = mapData;
    latestTelemetry = null;
    currentTargets = runData.targets || [];

    renderRunSummary(runData);
    renderTimeline(currentTimeline);
    renderTargets(currentTargets);
    renderOperatorQueue(runData.operator_queue || []);
    renderAlerts(currentTimeline);
    renderMap(mapData);
    syncReplayControls();
    renderReplayFrame(0);

    setStatus(`Loaded ${runId}`);
}

function startLiveLatest() {
    stopLive();
    pauseReplay();
    replayFrames = [];
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    liveWs = new WebSocket(`${protocol}//${location.host}/ws/live/latest`);
    liveWs.onopen = () => {
        setMode('live');
        document.getElementById('video-feed').src = '/video/live/latest';
        setStatus('Live mission connected');
    };
    liveWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === 'no_runs') {
            setStatus('Waiting for mission data...');
            return;
        }
        currentRunId = data.run_id;
        currentTargets = data.targets || [];
        latestMapData = data.map_layers || null;
        latestTelemetry = data.latest_uav_state || null;
        currentTimeline = data.timeline_tail || [];

        renderTargets(currentTargets);
        renderOperatorQueue(data.operator_queue || []);
        renderAlerts(currentTimeline);
        renderTimeline(currentTimeline);
        if (latestMapData) {
            latestMapData.uav_track = data.uav_track_tail || latestMapData.uav_track || [];
            renderMap(latestMapData);
        }
        renderTelemetry(latestTelemetry);
        renderLiveSummary(data.artifact_counts || {});
        renderVideoFrameMeta(data.latest_frame_name || '', latestTelemetry ? latestTelemetry.timestamp_utc : '');
        renderVideoFocus();
        setStatus(`Live ${currentRunId}`);
    };
    liveWs.onclose = () => {
        liveWs = null;
        if (isLive) {
            setMode('replay');
        }
    };
}

function stopLive() {
    if (liveWs) {
        liveWs.close();
        liveWs = null;
    }
    if (isLive) {
        setMode('replay');
        setStatus('Live stream stopped');
    }
}

function renderRunSummary(runData) {
    const artifacts = runData.artifacts || {};
    document.getElementById('target-summary').textContent = `${(runData.targets || []).length} tracks`;
    document.getElementById('map-summary').textContent = `${artifacts.observations_count || 0} observations`;
    document.getElementById('timeline-count').textContent = `${artifacts.events_count || 0} events`;
}

function renderLiveSummary(counts) {
    document.getElementById('target-summary').textContent = `${counts.tracks || 0} track updates`;
    document.getElementById('map-summary').textContent = `${counts.geo_observations || 0} observations`;
    document.getElementById('timeline-count').textContent = `${counts.events || 0} events`;
}

function renderTargets(targets) {
    const container = document.getElementById('targets-content');
    const moving = targets.filter(t => t.movement_state !== 'STATIONARY').length;
    const suppressed = targets.filter(t => t.suppressed).length;
    document.getElementById('target-summary').textContent = `${targets.length} tracks · ${moving} moving · ${suppressed} suppressed`;
    if (!targets.length) {
        container.innerHTML = '<div class="target-card">No targets yet.</div>';
        renderVideoFocus();
        return;
    }

    container.innerHTML = targets.map((target) => `
        <button class="target-card ${target.track_id === selectedTrackId ? 'selected' : ''} priority-${target.priority_level} ${target.suppressed ? 'suppressed' : ''}" data-track-id="${target.track_id}">
            <div class="target-top">
                <span class="target-id">${target.track_id}</span>
                <span class="status-pill status-${target.status}">${target.status}</span>
                <span class="priority-pill priority-${target.priority_level}">${target.priority_level}</span>
            </div>
            <div class="target-grid">
                <div><span class="datum-label">Class</span><span>${target.class_name}</span></div>
                <div><span class="datum-label">Confidence</span><span>${(target.confidence * 100).toFixed(0)}%</span></div>
                <div><span class="datum-label">Motion</span><span class="motion-${target.movement_state}">${target.movement_state}</span></div>
                <div><span class="datum-label">Speed</span><span>${target.average_speed_px_s.toFixed(1)} px/s</span></div>
                <div><span class="datum-label">Age</span><span>${target.age_frames}f</span></div>
                <div><span class="datum-label">Lost</span><span>${target.lost_frames}</span></div>
                <div><span class="datum-label">Reacquired</span><span>${target.reacquired_count}</span></div>
                <div><span class="datum-label">Last Seen</span><span>${formatTime(target.last_seen_utc)}</span></div>
            </div>
        </button>
    `).join('');

    container.querySelectorAll('.target-card').forEach((element) => {
        element.addEventListener('click', () => {
            selectedTrackId = element.dataset.trackId;
            renderTargets(currentTargets);
            renderMap(latestMapData);
            renderVideoFocus();
        });
    });

    renderVideoFocus();
}

function renderOperatorQueue(items) {
    const container = document.getElementById('queue-content');
    if (!items.length) {
        container.innerHTML = '<div class="queue-item">No operator tasks.</div>';
        return;
    }
    container.innerHTML = items.slice(-6).reverse().map((item) => `
        <div class="queue-item">
            <div class="queue-top">
                <strong>${item.track_id || item.observation_id}</strong>
                <span>${item.decision_state}</span>
            </div>
            <div>${item.policy_action || 'decision required'}</div>
            <small>${formatTime(item.requested_at)} ${item.reason || ''}</small>
        </div>
    `).join('');
}

function renderAlerts(events) {
    const container = document.getElementById('alerts-content');
    const alerts = (events || []).filter((event) => ['warning', 'error'].includes(event.severity)).slice(-6).reverse();
    if (!alerts.length) {
        container.innerHTML = '<div class="alert-item">No active alerts.</div>';
        return;
    }
    container.innerHTML = alerts.map((event) => `
        <div class="alert-item">
            <div class="alert-top">
                <strong>${event.type}</strong>
                <span>${event.severity}</span>
            </div>
            <div>${event.label || event.type}</div>
            <small>${formatTime(event.t)}</small>
        </div>
    `).join('');
}

function renderTimeline(events) {
    const container = document.getElementById('timeline-content');
    document.getElementById('timeline-count').textContent = `${events.length} events`;
    if (!events.length) {
        container.innerHTML = '<div class="timeline-item"><span class="time">--</span><span class="source">system</span><span class="label">No events.</span></div>';
        return;
    }
    container.innerHTML = events.slice(-30).map((event) => `
        <div class="timeline-item ${event.severity || 'info'}">
            <span class="time">${formatTime(event.t)}</span>
            <span class="source">${event.source}</span>
            <span class="label">${event.label || event.type}</span>
        </div>
    `).join('');
}

function renderTelemetry(state) {
    const container = document.getElementById('telemetry-content');
    if (!state) {
        container.innerHTML = '<div class="datum"><span class="datum-label">Telemetry</span><span class="datum-value">Unavailable</span></div>';
        return;
    }
    const cells = [
        ['Vehicle', state.vehicle_id || '-'],
        ['Mode', state.mode || '-'],
        ['Altitude', state.alt_m != null ? `${state.alt_m.toFixed(1)} m` : '-'],
        ['Speed', state.groundspeed_mps != null ? `${state.groundspeed_mps.toFixed(1)} m/s` : '-'],
        ['Heading', state.heading_deg != null ? `${state.heading_deg.toFixed(0)} deg` : '-'],
        ['Battery', state.battery_pct != null ? `${state.battery_pct.toFixed(0)}%` : '-'],
        ['State', state.armed ? 'ARMED' : 'DISARMED'],
        ['Updated', formatTime(state.timestamp_utc || '')],
    ];
    container.innerHTML = cells.map(([label, value]) => `
        <div class="datum">
            <span class="datum-label">${label}</span>
            <span class="datum-value">${value}</span>
        </div>
    `).join('');
}

function renderMap(data) {
    Object.values(layerGroups).forEach((group) => group.clearLayers());
    if (!data) {
        return;
    }

    const bounds = [];
    (data.waypoints || []).forEach((waypoint, index) => {
        layerGroups.waypoints.addLayer(L.circleMarker([waypoint.lat, waypoint.lon], {
            radius: 5, color: '#5f96bf', fillColor: '#5f96bf', fillOpacity: 0.9,
        }).bindTooltip(waypoint.label || `WP ${index + 1}`));
        bounds.push([waypoint.lat, waypoint.lon]);
    });

    const trackPoints = data.uav_track || [];
    if (trackPoints.length > 1) {
        layerGroups.track.addLayer(L.polyline(trackPoints.map((point) => [point.lat, point.lon]), {
            color: '#f2b544',
            weight: 3,
            opacity: 0.8,
        }));
    }

    const uavPositions = data.uav_positions || [];
    if (uavPositions.length) {
        const latest = uavPositions[uavPositions.length - 1];
        latestTelemetry = {
            vehicle_id: latest.vehicle_id,
            mode: latest.mode,
            alt_m: latest.alt_m,
            groundspeed_mps: latest.speed_mps,
            heading_deg: latest.heading,
            battery_pct: latest.battery_pct,
            armed: latest.armed,
            timestamp_utc: latest.timestamp,
        };
        renderTelemetry(latestTelemetry);
        layerGroups.uav.addLayer(L.circleMarker([latest.lat, latest.lon], {
            radius: 8,
            color: '#82d7a4',
            fillColor: '#82d7a4',
            fillOpacity: 1,
        }));
        bounds.push([latest.lat, latest.lon]);
    }

    (data.observations || []).forEach((obs) => {
        const isSelected = selectedTrackId && obs.track_id === selectedTrackId;
        const target = currentTargets.find(t => t.track_id === obs.track_id);
        const isSuppressed = target && target.suppressed;
        layerGroups.observations.addLayer(L.circleMarker([obs.lat, obs.lon], {
            radius: isSelected ? 10 : 6,
            color: obs.confirmed ? '#82d7a4' : '#f2b544',
            fillColor: obs.confirmed ? '#82d7a4' : '#f2b544',
            fillOpacity: isSuppressed ? 0.2 : 0.75,
        }).on('click', () => {
            selectedTrackId = obs.track_id;
            renderTargets(currentTargets);
            renderMap(latestMapData);
            renderVideoFocus();
        }));
        bounds.push([obs.lat, obs.lon]);
    });

    const selectedTarget = currentTargets.find((target) => target.track_id === selectedTrackId);
    if (selectedTarget && selectedTarget.observation_lat != null && selectedTarget.observation_lon != null) {
        layerGroups.highlight.addLayer(L.circle([selectedTarget.observation_lat, selectedTarget.observation_lon], {
            radius: 25,
            color: '#ffffff',
            weight: 1,
            fillOpacity: 0,
        }));
    }

    if (bounds.length && !isLive) {
        map.fitBounds(bounds, { padding: [30, 30] });
    }
}

function syncReplayControls() {
    const scrubber = document.getElementById('scrubber');
    scrubber.max = Math.max(replayFrames.length - 1, 0);
    scrubber.value = replayIndex;
    document.getElementById('scrubber-label').textContent = `${replayFrames.length ? replayIndex + 1 : 0} / ${replayFrames.length}`;
}

function renderReplayFrame(index) {
    if (!replayFrames.length) {
        document.getElementById('video-feed').removeAttribute('src');
        renderTelemetry(null);
        renderVideoFrameMeta('', '');
        renderVideoFocus();
        return;
    }

    replayIndex = Math.min(index, replayFrames.length - 1);
    const frame = replayFrames[replayIndex];
    currentTargets = frame.targets || [];
    renderTargets(currentTargets);
    renderTelemetry(frame.vehicle_state || latestTelemetry);
    renderVideoFrameMeta(frame.frame_name, frame.timestamp_utc);
    if (latestMapData) {
        renderMap(filterMapForFrame(latestMapData, frame.frame_id));
    }
    document.getElementById('video-feed').src = `${API}/api/runs/${currentRunId}/frames/${frame.frame_name}`;
    document.getElementById('scrubber').value = replayIndex;
    document.getElementById('scrubber-label').textContent = `${replayIndex + 1} / ${replayFrames.length}`;
}

function renderVideoFrameMeta(frameName, timestamp) {
    document.getElementById('video-frame-label').textContent = frameName || 'Frame --';
    document.getElementById('video-time-label').textContent = timestamp ? formatTime(timestamp) : '--';
}

function startReplay() {
    if (!replayFrames.length) {
        return;
    }
    pauseReplay();
    replayTimer = window.setInterval(() => {
        replayIndex = (replayIndex + 1) % replayFrames.length;
        renderReplayFrame(replayIndex);
    }, Math.max(120, 800 / replaySpeed));
}

function pauseReplay() {
    if (replayTimer) {
        window.clearInterval(replayTimer);
        replayTimer = null;
    }
}

function renderVideoFocus() {
    const overlay = document.getElementById('video-overlay');
    overlay.innerHTML = '';
    if (!selectedTrackId) {
        return;
    }
    const target = currentTargets.find((item) => item.track_id === selectedTrackId);
    if (!target || !target.bbox) {
        return;
    }
    const box = document.createElement('div');
    box.className = 'focus-box';
    box.style.left = `${(target.bbox.x1 / videoFrameSize.width) * 100}%`;
    box.style.top = `${(target.bbox.y1 / videoFrameSize.height) * 100}%`;
    box.style.width = `${((target.bbox.x2 - target.bbox.x1) / videoFrameSize.width) * 100}%`;
    box.style.height = `${((target.bbox.y2 - target.bbox.y1) / videoFrameSize.height) * 100}%`;
    overlay.appendChild(box);
}

function formatTime(timestamp) {
    if (!timestamp) {
        return '--';
    }
    if (timestamp.length >= 19) {
        return timestamp.substring(11, 19);
    }
    return timestamp;
}

function filterMapForFrame(mapData, frameId) {
    return {
        ...mapData,
        observations: (mapData.observations || []).filter((item) => item.frame_id == null || item.frame_id <= frameId),
        uav_positions: (mapData.uav_positions || []).filter((item) => item.frame_id == null || item.frame_id <= frameId),
        uav_track: (mapData.uav_track || []).filter((item) => item.frame_id == null || item.frame_id <= frameId),
    };
}
