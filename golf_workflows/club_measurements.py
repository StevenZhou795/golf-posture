"""Angle-only shaft measurements for frames where the club detector is unsure.

Two independent cues, both measured as the direction from the hands toward
the clubhead, so there is no grip/hosel ambiguity:

- Motion streak: at speed the shaft is a blur. Pixels that differ from both the
  previous and the next frame (moving at this frame, not revealed background)
  are collected in an annulus around the hands; the dominant direction of their
  angle histogram is the shaft.
- Club mask: the golf-club segmentation model's mask nearest the hands; its
  principal axis, oriented away from the hands.

Each returns (angle, sigma radians, source) for the club tracker, which gates
them against its prediction.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

STREAK_SIGMA_DEG = 12.0
MASK_SIGMA_DEG = 6.0
INNER_RADIUS = 0.35
OUTER_RADIUS = 1.15
MIN_STREAK_PIXELS = 40
MIN_STREAK_SUPPORT = 0.3
MIN_RADIAL_SPREAD = 0.12
MIN_MASK_ELONGATION = 4.0
MAX_MASK_GAP = 0.4
MASK_CONFIDENCE = 0.25


def streak_angle(previous: np.ndarray, current: np.ndarray, following: np.ndarray, hands: np.ndarray,
                 length: float) -> tuple[float, dict] | None:
    """Direction of the motion blur around the hands, in image radians (y down)."""
    height, width = current.shape[:2]
    radius = OUTER_RADIUS * length
    x0, x1 = int(max(0, hands[0] - radius)), int(min(width, hands[0] + radius + 1))
    y0, y1 = int(max(0, hands[1] - radius)), int(min(height, hands[1] + radius + 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crops = [cv2.GaussianBlur(image[y0:y1, x0:x1], (3, 3), 0).astype(np.int16) for image in (previous, current, following)]
    moving = np.minimum(np.abs(crops[1] - crops[0]), np.abs(crops[2] - crops[1])).astype(np.float64)
    threshold = max(20.0, 3.0 * float(np.median(moving)))
    ys, xs = np.nonzero(moving > threshold)
    if len(xs) < MIN_STREAK_PIXELS:
        return None
    dx, dy = xs + x0 - hands[0], ys + y0 - hands[1]
    r = np.hypot(dx, dy)
    ring = (r >= INNER_RADIUS * length) & (r <= radius)
    if ring.sum() < MIN_STREAK_PIXELS:
        return None
    weights = moving[ys[ring], xs[ring]]
    angles = np.arctan2(dy[ring], dx[ring])
    bins = 72
    histogram, edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=weights)
    smoothed = histogram + 0.5 * (np.roll(histogram, 1) + np.roll(histogram, -1))
    peak = int(np.argmax(smoothed))
    centre = 0.5 * (edges[peak] + edges[peak + 1])
    near = np.abs((angles - centre + np.pi) % (2 * np.pi) - np.pi) <= np.radians(10)
    support = float(weights[near].sum() / max(weights.sum(), 1e-9))
    spread = float(np.std(r[ring][near]) / length) if near.sum() > 1 else 0.0
    if support < MIN_STREAK_SUPPORT or spread < MIN_RADIAL_SPREAD:
        return None
    # The shaft line does not pass through the wrist midpoint, so fit the line
    # through the streak pixels and orient it away from the hands.
    xy = np.column_stack([dx[ring][near], dy[ring][near]])
    w = weights[near] / weights[near].sum()
    centre_xy = (w[:, None] * xy).sum(axis=0)
    spread_matrix = (w[:, None] * (xy - centre_xy)).T @ (xy - centre_xy)
    values, vectors = np.linalg.eigh(spread_matrix)
    axis = vectors[:, 1]
    if np.dot(axis, centre_xy) < 0:
        axis = -axis
    angle = float(np.arctan2(axis[1], axis[0]))
    return angle, {"pixels": int(near.sum()), "support": round(support, 3), "radial_spread": round(spread, 3)}


def mask_angle(result, hands: np.ndarray, length: float) -> tuple[float, dict] | None:
    """Principal axis of the club mask nearest the hands, oriented away from the hands."""
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None:
        return None
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(masks.xy))
    best = None
    for polygon, score in zip(masks.xy, scores):
        if len(polygon) < 5 or score < MASK_CONFIDENCE:
            continue
        points = np.asarray(polygon, dtype=np.float64)
        gap = float(np.min(np.linalg.norm(points - hands, axis=1)))
        if gap > MAX_MASK_GAP * length:
            continue
        centre = points.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points - centre).T))
        if values[0] <= 0 or np.sqrt(values[1] / values[0]) < MIN_MASK_ELONGATION:
            continue
        axis = vectors[:, 1]
        projection = (points - centre) @ axis
        ends = centre + np.outer([projection.min(), projection.max()], axis)
        far = ends[int(np.argmax(np.linalg.norm(ends - hands, axis=1)))]
        direction = far - hands
        candidate = (float(score), float(np.arctan2(direction[1], direction[0])), gap)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], {"score": round(best[0], 3), "gap_over_length": round(best[2] / length, 3)}


def shaft_angle_measurements(
    video: Path,
    frames: set[int],
    hands: np.ndarray,
    length_px: float,
    mask_model=None,
) -> tuple[dict[int, list[tuple[float, float, str]]], dict]:
    """Streak and mask angles for `frames`, reading the video once."""
    measurements: dict[int, list[tuple[float, float, str]]] = {}
    counts = {"requested": len(frames), "streak": 0, "mask": 0}
    if not frames or not np.isfinite(length_px) or length_px <= 0:
        return measurements, counts
    capture = cv2.VideoCapture(str(video))
    window: list[tuple[int, np.ndarray, np.ndarray]] = []
    index = 0
    last = max(frames) + 1
    try:
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            window.append((index, frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            window = window[-3:]
            if len(window) == 3:
                target, colour, gray = window[1]
                if target in frames and np.isfinite(hands[target]).all():
                    found = []
                    streak = streak_angle(window[0][2], gray, window[2][2], hands[target], length_px)
                    if streak is not None:
                        found.append((streak[0], np.radians(STREAK_SIGMA_DEG), "streak"))
                        counts["streak"] += 1
                    if mask_model is not None:
                        result = mask_model.predict(colour, verbose=False, conf=MASK_CONFIDENCE)[0]
                        mask = mask_angle(result, hands[target], length_px)
                        if mask is not None:
                            found.append((mask[0], np.radians(MASK_SIGMA_DEG), "mask"))
                            counts["mask"] += 1
                    if found:
                        measurements[target] = found
            index += 1
    finally:
        capture.release()
    return measurements, counts
