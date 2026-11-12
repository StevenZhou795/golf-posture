"""Visibility gating, time-based gap filling, smoothing and 3D bone-length consistency."""
import numpy as np
from scipy.signal import savgol_filter

from blazepose_workflow.landmarks import BONES, J

VISIBLE = 0.5
MAX_GAP_SECONDS = 0.3
SMOOTH_SECONDS = 0.1


def gate(points: np.ndarray, visibility: np.ndarray, threshold: float = VISIBLE) -> np.ndarray:
    """Coordinates with landmarks below `threshold` visibility set to NaN."""
    gated = points.astype(np.float64).copy()
    gated[~(visibility >= threshold)] = np.nan
    return gated


def fill_gaps(points: np.ndarray, times_ms: np.ndarray, max_gap_seconds: float = MAX_GAP_SECONDS) -> np.ndarray:
    """Linear interpolation in time across gaps no longer than `max_gap_seconds`.

    Gaps at the clip ends and longer gaps stay NaN, so nothing downstream is
    chosen from invented poses.
    """
    filled = points.copy()
    times = np.asarray(times_ms, float)
    flat = filled.reshape(len(filled), -1)
    for column in range(flat.shape[1]):
        series = flat[:, column]
        good = np.isfinite(series)
        if good.sum() < 2:
            continue
        index = np.flatnonzero(good)
        for a, b in zip(index[:-1], index[1:]):
            if b - a > 1 and times[b] - times[a] <= max_gap_seconds * 1000 + 1e-6:
                series[a + 1:b] = np.interp(times[a + 1:b], [times[a], times[b]], [series[a], series[b]])
    return filled


def smooth(points: np.ndarray, times_ms: np.ndarray, seconds: float = SMOOTH_SECONDS) -> np.ndarray:
    """Savitzky-Golay (quadratic) smoothing over ~`seconds`, per finite run."""
    if len(times_ms) < 2:
        return points.copy()
    step = float(np.median(np.diff(times_ms))) / 1000 or 1 / 30
    window = int(round(seconds / step)) | 1
    if window < 5:
        return points.copy()
    smoothed = points.copy()
    flat = smoothed.reshape(len(smoothed), -1)
    for column in range(flat.shape[1]):
        series = flat[:, column]
        good = np.isfinite(series)
        edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= window:
                series[start:stop] = savgol_filter(series[start:stop], window, 2)
    return smoothed


def bone_lengths(world: np.ndarray) -> np.ndarray:
    """(n, bones) length of each bone in BONES."""
    return np.stack([np.linalg.norm(world[:, b] - world[:, a], axis=-1) for a, b in BONES], axis=1)


def normalize_bones(world: np.ndarray) -> np.ndarray:
    """Rescale each bone to its clip median length, keeping the hip centre at the origin.

    GHUM world landmarks are fitted per frame, so the same limb changes length
    from frame to frame. Joints are re-placed outward from the hips along each
    bone's direction, parents before children.
    """
    result = world.copy()
    target = np.nanmedian(bone_lengths(world), axis=0)
    left, right = J["left_hip"], J["right_hip"]
    hip_bone = BONES.index((left, right))
    centre = (world[:, left] + world[:, right]) / 2
    half = (world[:, left] - world[:, right]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        half = half / np.linalg.norm(half, axis=1, keepdims=True) * target[hip_bone] / 2
    result[:, left], result[:, right] = centre + half, centre - half
    order = _bone_order()
    for frame in range(len(world)):
        for bone, (parent, child) in order:
            vector = world[frame, child] - world[frame, parent]
            length = float(np.linalg.norm(vector))
            if not np.isfinite(length) or length < 1e-9 or not np.isfinite(target[bone]):
                continue
            result[frame, child] = result[frame, parent] + vector / length * target[bone]
    return result


def _bone_order() -> list[tuple[int, tuple[int, int]]]:
    """BONES oriented away from the hips, in breadth-first order; cross bones are skipped."""
    roots = {J["left_hip"], J["right_hip"]}
    placed, order = set(roots), []
    remaining = list(enumerate(BONES))
    while remaining:
        progress = False
        for item in list(remaining):
            bone, (a, b) = item
            if a in placed and b in placed:
                remaining.remove(item)
                progress = True
            elif a in placed or b in placed:
                parent, child = (a, b) if a in placed else (b, a)
                order.append((bone, (parent, child)))
                placed.add(child)
                remaining.remove(item)
                progress = True
        if not progress:
            break
    return order
