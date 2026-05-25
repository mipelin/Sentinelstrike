"""Detection-to-track association using Hungarian algorithm + cascade matching.

Implements 4+1-stage cascade matching (BoT-SORT style) without scipy dependency.
Stage 5 uses IdentityMemory for cross-lifecycle re-identification.
The Hungarian algorithm is implemented via the shortest augmenting path method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .appearance import AppearanceGallery, cosine_distance, extract_features

if TYPE_CHECKING:
    from .reid import IdentityMemory


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) tuples."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Solve linear assignment problem using shortest augmenting path.

    Given an n×m cost matrix (n detections, m tracks), returns the optimal
    assignment as a list of (detection_idx, track_idx) pairs.

    Based on the Jonker-Volgenant algorithm (LAPJV).
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []

    # Pad to square if needed
    dim = max(n_rows, n_cols)
    if n_rows != n_cols:
        padded = np.full((dim, dim), 1e6, dtype=np.float64)
        padded[:n_rows, :n_cols] = cost
        cost = padded

    n = dim
    INF = 1e18

    # cost modulation: subtract row/col mins for tighter bounds
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)   # p[j] = row assigned to column j
    way = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    result = []
    for j in range(1, n + 1):
        if p[j] >= 1 and p[j] <= n_rows and j <= n_cols:
            result.append((p[j] - 1, j - 1))
    return result


def iou_cost_matrix(
    detections: list[tuple[float, float, float, float]],
    tracks: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Compute IoU-based cost matrix. Lower = better match.

    Returns shape (len(detections), len(tracks)).
    Cost = 1 - IoU, so IoU=1 → cost=0 (perfect), IoU=0 → cost=1 (no overlap).
    """
    n = len(detections)
    m = len(tracks)
    cost = np.ones((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            cost[i, j] = 1.0 - _iou(detections[i], tracks[j])
    return cost


def appearance_cost_matrix(
    det_features: list[np.ndarray | None],
    galleries: list[AppearanceGallery],
) -> np.ndarray:
    """Compute appearance-based cost matrix. Lower = better match.

    Returns shape (len(det_features), len(galleries)).
    """
    n = len(det_features)
    m = len(galleries)
    cost = np.ones((n, m), dtype=np.float32)
    for i in range(n):
        if det_features[i] is None:
            continue
        for j in range(m):
            cost[i, j] = galleries[j].match_distance(det_features[i])
    return cost


def combined_cost_matrix(
    detections: list[tuple[float, float, float, float]],
    tracks: list[tuple[float, float, float, float]],
    det_features: list[np.ndarray | None],
    galleries: list[AppearanceGallery],
    iou_weight: float = 0.7,
    appearance_weight: float = 0.3,
    identity_stabilities: list[float] | None = None,
) -> np.ndarray:
    """Weighted combination of IoU and appearance costs.

    When identity_stabilities is provided, tracks with high stability (>0.8)
    get boosted appearance weight for more reliable matching.
    """
    iou_cost = iou_cost_matrix(detections, tracks)
    app_cost = appearance_cost_matrix(det_features, galleries)

    # If no appearance features available, fall back to IoU only
    has_app = any(f is not None for f in det_features) and any(g.features for g in galleries)
    if not has_app:
        return iou_cost

    # Dynamic weights: boost appearance for stable identities
    if identity_stabilities and len(identity_stabilities) == len(galleries):
        result = np.ones_like(iou_cost)
        n_det = len(detections)
        for j, gallery in enumerate(galleries):
            stab = identity_stabilities[j]
            aw = appearance_weight * (1.3 if stab > 0.8 else 1.0)
            iw = iou_weight
            total = aw + iw
            for i in range(n_det):
                result[i, j] = (iw / total) * iou_cost[i, j] + (aw / total) * app_cost[i, j]
        return result

    return iou_weight * iou_cost + appearance_weight * app_cost


def match_detections_to_tracks(
    det_indices: list[int],
    track_indices: list[int],
    det_bboxes: list[tuple[float, float, float, float]],
    track_bboxes: list[tuple[float, float, float, float]],
    det_features: list[np.ndarray | None],
    galleries: list[AppearanceGallery],
    gate: float = 0.8,
    iou_weight: float = 0.7,
    appearance_weight: float = 0.3,
    iou_only: bool = False,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to tracks using Hungarian algorithm with gating.

    Args:
        det_indices: Original indices of detections being considered
        track_indices: Original indices of tracks being considered
        det_bboxes: Bboxes for these detections (same order as det_indices)
        track_bboxes: Bboxes for these tracks (same order as track_indices)
        det_features: Appearance features for detections
        galleries: Appearance galleries for tracks
        gate: Maximum cost threshold for accepting a match
        iou_weight/appearance_weight: Cost combination weights
        iou_only: If True, use IoU cost only (no appearance)

    Returns:
        (matched_pairs, unmatched_det_indices, unmatched_track_indices)
        where matched_pairs contains (original_det_idx, original_track_idx)
    """
    if not det_indices or not track_indices:
        return [], list(det_indices), list(track_indices)

    if iou_only:
        cost = iou_cost_matrix(det_bboxes, track_bboxes)
    else:
        cost = combined_cost_matrix(
            det_bboxes, track_bboxes,
            det_features, galleries,
            iou_weight, appearance_weight,
        )

    assignments = _hungarian(cost)

    matched = []
    used_dets: set[int] = set()
    used_tracks: set[int] = set()

    for di, ti in assignments:
        if cost[di, ti] > gate:
            continue
        matched.append((det_indices[di], track_indices[ti]))
        used_dets.add(di)
        used_tracks.add(ti)

    unmatched_dets = [det_indices[i] for i in range(len(det_indices)) if i not in used_dets]
    unmatched_tracks = [track_indices[i] for i in range(len(track_indices)) if i not in used_tracks]

    return matched, unmatched_dets, unmatched_tracks


def cascade_match(
    detections: list[dict],
    tracks: list,
    high_conf_threshold: float = 0.5,
    iou_gate: float = 0.8,
    lost_iou_gate: float = 0.9,
    iou_weight: float = 0.7,
    appearance_weight: float = 0.3,
    identity_memory: IdentityMemory | None = None,
    reid_threshold: float = 0.6,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """4+1-stage cascade matching (BoT-SORT style).

    Stage 1: High-confidence dets → confirmed tracks (IoU + appearance)
    Stage 2: Low-confidence dets → remaining confirmed tracks (IoU only)
    Stage 3: Unmatched dets → tentative tracks (IoU only)
    Stage 4: Unmatched dets → lost tracks (IoU + appearance, wider gate)
    Stage 5: Unmatched dets → identity memory (cross-lifecycle Re-ID)

    Args:
        detections: List of detection dicts with 'bbox', 'confidence', 'class'
        tracks: List of ISRTrackState objects
        high_conf_threshold: Confidence above which detection is "high confidence"
        iou_gate: Cost gate for stages 1-2
        lost_iou_gate: Wider cost gate for stage 4 (lost track recovery)
        iou_weight/appearance_weight: Cost combination weights

    Returns:
        (matched_pairs, unmatched_det_indices, unmatched_track_indices)
    """
    confirmed_idx = [i for i, t in enumerate(tracks) if t.state == "CONFIRMED"]
    tentative_idx = [i for i, t in enumerate(tracks) if t.state == "TENTATIVE"]
    lost_idx = [i for i, t in enumerate(tracks) if t.state == "LOST"]

    # Partition detections by confidence
    high_det_idx = [i for i, d in enumerate(detections) if d["confidence"] >= high_conf_threshold]
    low_det_idx = [i for i, d in enumerate(detections) if d["confidence"] < high_conf_threshold]

    all_matched: list[tuple[int, int]] = []
    matched_det_set: set[int] = set()
    matched_trk_set: set[int] = set()

    def _add_matches(pairs: list[tuple[int, int]]) -> None:
        for di, ti in pairs:
            if di not in matched_det_set and ti not in matched_trk_set:
                all_matched.append((di, ti))
                matched_det_set.add(di)
                matched_trk_set.add(ti)

    # Stage 1: high-confidence → confirmed tracks (IoU + appearance)
    available_confirmed = [i for i in confirmed_idx if i not in matched_trk_set]
    if high_det_idx and available_confirmed:
        det_bboxes = [tuple(detections[i]["bbox"]) for i in high_det_idx]
        trk_bboxes = [tracks[i].predicted_bbox for i in available_confirmed]
        det_feats = [detections[i].get("_feature") for i in high_det_idx]
        galleries = [tracks[i].appearance_gallery for i in available_confirmed]

        m1, _, _ = match_detections_to_tracks(
            high_det_idx, available_confirmed, det_bboxes, trk_bboxes,
            det_feats, galleries,
            gate=iou_gate, iou_weight=iou_weight,
            appearance_weight=appearance_weight,
        )
        _add_matches(m1)

    # Stage 2: low-confidence → remaining confirmed tracks (IoU only)
    remaining_confirmed = [i for i in confirmed_idx if i not in matched_trk_set]
    if low_det_idx and remaining_confirmed:
        det_bboxes = [tuple(detections[i]["bbox"]) for i in low_det_idx]
        trk_bboxes = [tracks[i].predicted_bbox for i in remaining_confirmed]

        m2, _, _ = match_detections_to_tracks(
            low_det_idx, remaining_confirmed, det_bboxes, trk_bboxes,
            [], [], gate=iou_gate, iou_only=True,
        )
        _add_matches(m2)

    # Stage 3: unmatched detections → tentative tracks (IoU only)
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_set]
    available_tentative = [i for i in tentative_idx if i not in matched_trk_set]
    if unmatched_dets and available_tentative:
        det_bboxes = [tuple(detections[i]["bbox"]) for i in unmatched_dets]
        trk_bboxes = [tracks[i].predicted_bbox for i in available_tentative]

        m3, _, _ = match_detections_to_tracks(
            unmatched_dets, available_tentative, det_bboxes, trk_bboxes,
            [], [], gate=iou_gate, iou_only=True,
        )
        _add_matches(m3)

    # Stage 4: still unmatched detections → lost tracks (IoU + appearance, wider gate)
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_set]
    available_lost = [i for i in lost_idx if i not in matched_trk_set]
    if unmatched_dets and available_lost:
        det_bboxes = [tuple(detections[i]["bbox"]) for i in unmatched_dets]
        trk_bboxes = [tracks[i].predicted_bbox for i in available_lost]
        det_feats = [detections[i].get("_feature") for i in unmatched_dets]
        galleries = [tracks[i].appearance_gallery for i in available_lost]

        m4, truly_unmatched, _ = match_detections_to_tracks(
            unmatched_dets, available_lost, det_bboxes, trk_bboxes,
            det_feats, galleries,
            gate=lost_iou_gate, iou_weight=iou_weight,
            appearance_weight=appearance_weight,
        )
        _add_matches(m4)
    else:
        truly_unmatched = [i for i in range(len(detections)) if i not in matched_det_set]

    # Stage 5: truly unmatched detections → identity memory (cross-lifecycle Re-ID)
    if truly_unmatched and identity_memory is not None:
        for di in truly_unmatched:
            det = detections[di]
            if det.get("_feature") is None:
                continue
            cx = (det["_bbox"][0] + det["_bbox"][2]) / 2.0
            cy = (det["_bbox"][1] + det["_bbox"][3]) / 2.0
            matched_id, score, breakdown = identity_memory.find_match(
                feature=det["_feature"],
                cx=cx, cy=cy,
                heading=0.0,
                frame_w=frame_w, frame_h=frame_h,
                class_name=det["class"],
                threshold=reid_threshold,
            )
            if matched_id is not None:
                det["_reid_match"] = {
                    "original_track_id": matched_id,
                    "score": score,
                    "breakdown": breakdown,
                }

    # Find unmatched tracks (all states that weren't matched)
    all_track_idx = confirmed_idx + tentative_idx + lost_idx
    unmatched_tracks = [i for i in all_track_idx if i not in matched_trk_set]

    return all_matched, truly_unmatched, unmatched_tracks
