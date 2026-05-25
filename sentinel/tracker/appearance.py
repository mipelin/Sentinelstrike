"""Spatial-color-grid appearance features for aerial ISR tracking.

Extracts discriminative body-region appearance features for re-identification:
- Upper body color (shirt/torso region)
- Lower body color (pants/legs region)
- Height/width ratio
- Dominant colors per region
- Global HSV histogram

Feature vector is 149-dim:
  - 96-dim: 2x2 spatial grid, per-cell 16-bin H + 8-bin S histogram
  - 5-dim: global HSV stats
  - 16-dim: upper body color histogram (H)
  - 8-dim: upper body color histogram (S)
  - 8-dim: lower body color histogram (H)
  - 4-dim: lower body color histogram (S)
  - 4-dim: shape features (aspect ratio, fill ratio, relative height, relative width)
  - 8-dim: dominant color bins (top-4 H bins for upper, top-4 H bins for lower)
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

FEATURE_DIM = 149


def extract_features(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Extract body-region appearance features from a bbox crop.

    Returns L2-normalized feature vector or None if crop too small.
    """
    if not _CV2:
        return None

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    cell_h = bh // 2
    cell_w = bw // 2

    if cell_h < 2 or cell_w < 2:
        return _fallback_features(hsv, bw, bh)

    # --- 96-dim: 2x2 spatial grid ---
    grid_features = np.zeros(96, dtype=np.float32)
    idx = 0
    for row in range(2):
        for col in range(2):
            ry = row * cell_h
            rx = col * cell_w
            cell = hsv[ry:ry + cell_h, rx:rx + cell_w]

            h_hist = cv2.calcHist([cell], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([cell], [1], None, [8], [0, 256]).flatten()
            cell_feat = np.concatenate([h_hist, s_hist]).astype(np.float32)
            grid_features[idx:idx + 24] = cell_feat
            idx += 24

    # --- 5-dim: global HSV stats ---
    stats = np.array([
        float(np.mean(hsv[:, :, 0])),
        float(np.mean(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 2])),
        float(np.std(hsv[:, :, 0])),
        float(np.std(hsv[:, :, 1])),
    ], dtype=np.float32)

    # --- Upper body (top 50% of crop) ---
    upper = hsv[:cell_h, :]
    upper_h = cv2.calcHist([upper], [0], None, [16], [0, 180]).flatten().astype(np.float32)
    upper_s = cv2.calcHist([upper], [1], None, [8], [0, 256]).flatten().astype(np.float32)

    # --- Lower body (bottom 50% of crop) ---
    lower = hsv[cell_h:, :]
    lower_h = cv2.calcHist([lower], [0], None, [8], [0, 180]).flatten().astype(np.float32)
    lower_s = cv2.calcHist([lower], [1], None, [4], [0, 256]).flatten().astype(np.float32)

    # --- 4-dim: shape features ---
    aspect_ratio = bh / max(bw, 1)
    fill_area = bw * bh
    # Approximate fill ratio from edge detection
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nonzero = cv2.countNonZero(binary)
    fill_ratio = nonzero / max(fill_area, 1)
    shape = np.array([aspect_ratio, fill_ratio, bh / max(h, 1), bw / max(w, 1)], dtype=np.float32)

    # --- 8-dim: dominant color bins ---
    upper_top4 = np.argsort(upper_h)[-4:].astype(np.float32) / 16.0
    lower_top4 = np.argsort(lower_h)[-4:].astype(np.float32) / 8.0
    dominant = np.concatenate([upper_top4, lower_top4])

    # --- Concatenate ---
    feature = np.concatenate([
        grid_features,   # 96
        stats,           # 5
        upper_h,         # 16
        upper_s,         # 8
        lower_h,         # 8
        lower_s,         # 4
        shape,           # 4
        dominant,        # 8
    ]).astype(np.float32)

    # Trim/pad to exact FEATURE_DIM
    if len(feature) > FEATURE_DIM:
        feature = feature[:FEATURE_DIM]
    elif len(feature) < FEATURE_DIM:
        padded = np.zeros(FEATURE_DIM, dtype=np.float32)
        padded[:len(feature)] = feature
        feature = padded

    norm = np.linalg.norm(feature)
    if norm < 1e-8:
        return None
    feature /= norm
    return feature


def _fallback_features(hsv: np.ndarray, bw: int, bh: int) -> np.ndarray | None:
    """Compute features without spatial grid for very small crops."""
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist]).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-8:
        return None
    feat /= norm
    padded = np.zeros(FEATURE_DIM, dtype=np.float32)
    padded[:len(feat)] = feat
    # Add shape
    padded[101] = bh / max(bw, 1)
    return padded


def extract_identity_descriptor(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict:
    """Extract human-readable identity descriptor for overlay display.

    Returns dict with upper_color, lower_color, dominant_colors, aspect_ratio.
    """
    if not _CV2:
        return {}

    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return {}

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mid = bh // 2

    # Upper body mean color
    upper = hsv[:mid, :]
    upper_mean = np.mean(upper, axis=(0, 1))
    upper_desc = _hsv_to_name(upper_mean[0], upper_mean[1], upper_mean[2])

    # Lower body mean color
    lower = hsv[mid:, :]
    lower_mean = np.mean(lower, axis=(0, 1))
    lower_desc = _hsv_to_name(lower_mean[0], lower_mean[1], lower_mean[2])

    return {
        "upper_color": upper_desc,
        "lower_color": lower_desc,
        "aspect_ratio": round(bh / max(bw, 1), 2),
        "upper_hsv": [round(v, 1) for v in upper_mean.tolist()],
        "lower_hsv": [round(v, 1) for v in lower_mean.tolist()],
    }


def _hsv_to_name(h: float, s: float, v: float) -> str:
    """Convert HSV values to a human-readable color name."""
    if v < 40:
        return "black"
    if s < 40:
        if v > 180:
            return "white"
        return "gray"
    # Hue-based naming
    if h < 15 or h >= 165:
        name = "red"
    elif h < 35:
        name = "orange"
    elif h < 70:
        name = "yellow"
    elif h < 85:
        name = "lime"
    elif h < 150:
        name = "green"
    elif h < 170:
        name = "teal"
    elif h < 195:
        name = "blue"
    elif h < 260:
        name = "purple"
    elif h < 290:
        name = "pink"
    else:
        name = "red"

    if v < 100:
        name = "dark_" + name
    elif s < 80:
        name = "pale_" + name
    return name


def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine distance between two feature vectors. 0 = identical, 1 = orthogonal."""
    if a is None or b is None:
        return 1.0
    a = a.flatten()
    b = b.flatten()
    if len(a) != len(b):
        return 1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - sim


def batch_cosine_distance(
    features: list[np.ndarray | None],
    gallery: list[np.ndarray | None],
) -> np.ndarray:
    """Compute pairwise cosine distance matrix. Returns (n, m)."""
    n = len(features)
    m = len(gallery)
    dist = np.ones((n, m), dtype=np.float32)
    for i in range(n):
        if features[i] is None:
            continue
        for j in range(m):
            if gallery[j] is None:
                continue
            dist[i, j] = cosine_distance(features[i], gallery[j])
    return dist


class AppearanceGallery:
    """Manages appearance feature history for a single track.

    Keeps up to `max_size` feature snapshots. Matching uses minimum
    distance to any snapshot (best-match policy).
    """

    def __init__(self, max_size: int = 15) -> None:
        self._features: list[np.ndarray] = []
        self._max_size = max_size

    def update(self, feature: np.ndarray | None, ema_alpha: float = 0.2) -> None:
        """Add or EMA-update the latest feature into the gallery."""
        if feature is None:
            return

        if not self._features:
            self._features.append(feature.copy())
            return

        # EMA update the most recent feature
        latest = self._features[-1]
        updated = ema_alpha * feature + (1.0 - ema_alpha) * latest
        norm = np.linalg.norm(updated)
        if norm > 1e-8:
            updated /= norm
        self._features[-1] = updated

        # Store raw features periodically (every 3rd update)
        if len(self._features) < self._max_size:
            self._features.append(feature.copy())
        elif len(self._features) > self._max_size:
            self._features = self._features[-self._max_size:]

    def match_distance(self, feature: np.ndarray | None) -> float:
        """Best-match cosine distance with temporal weighting. 0 = perfect match."""
        if feature is None or not self._features:
            return 1.0
        best = 1.0
        n = len(self._features)
        for idx, gf in enumerate(self._features):
            d = cosine_distance(feature, gf)
            # Temporal weight: recent features (higher index) weighted more
            age = n - 1 - idx  # 0 = most recent
            weight = 0.9 ** age
            weighted_d = d * weight + (1.0 - weight) * 1.0
            if weighted_d < best:
                best = weighted_d
                if best < 0.01:
                    break
        return best

    def stability_score(self) -> float:
        """Measure consistency of stored features. 1.0 = all identical."""
        if len(self._features) < 2:
            return 0.0
        distances = []
        n = len(self._features)
        step = max(1, n // 5)
        for i in range(0, n - 1, step):
            j = min(i + step, n - 1)
            distances.append(cosine_distance(self._features[i], self._features[j]))
        mean_dist = sum(distances) / len(distances)
        return max(0.0, 1.0 - mean_dist)

    @property
    def features(self) -> list[np.ndarray | None]:
        return list(self._features)

    @property
    def latest(self) -> np.ndarray | None:
        return self._features[-1] if self._features else None
