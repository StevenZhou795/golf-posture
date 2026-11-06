"""2D club tracker: a rigid stick anchored at the hands.

State per frame is the shaft angle (grip to hosel, image radians) and its
angular speed. Detections and angle-only measurements (motion streak, club
mask) update a constant-angular-velocity Kalman filter; a Rauch-Tung-Striebel
pass smooths it. The projected length and the grip offset from the hands are
interpolated from accepted detections, because foreshortening changes them
slowly and independently of the angle.

Grip/hosel swaps are fixed by geometry first: the grip is the end at the
hands. Only when both ends are about as far from the hands (a foreshortened
club) may continuity reverse a detection whose reversed direction fits the
prediction. Confident detections (>= 0.5) are always accepted and restart the
state when they disagree with it; low-confidence detections outside the gate
are rejected and lose their valid flag downstream. The smoothed angle variance gives a
per-frame uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SHAFT, HOSEL = 17, 18
# Angle noise of a detection at confidence 1, and image noise per endpoint.
DETECTION_SIGMA_DEG = 3.0
ENDPOINT_SIGMA_PX = 6.0
# Angular acceleration noise at 30 fps (scaled by (30/fps)^2).
ACCELERATION_SIGMA_DEG = 15.0
GATE_SIGMAS = 3.0
ALWAYS_ACCEPT_DEG = 20.0
# Consecutive rejected confident detections that mean the track was lost.
REINIT_AFTER = 4
# Detections at or above this confidence are never rejected or reversed by continuity.
REINIT_CONFIDENCE = 0.5
# Ends differing by at least this fraction of the club length in distance to the
# hands fix the grip/hosel direction geometrically.
ORIENTED_FRACTION = 0.25


@dataclass
class ClubTrack:
    points: np.ndarray            # (n, 2, 2) grip and hosel, every frame
    theta: np.ndarray             # (n,) smoothed angle, radians, unwrapped
    sigma_deg: np.ndarray         # (n,) smoothed angle standard deviation
    accepted: np.ndarray          # (n,) detection accepted
    rejected: np.ndarray          # (n,) detection present but outside the gate
    swapped: np.ndarray           # (n,) detection accepted with grip and hosel exchanged
    measurement_used: np.ndarray  # (n,) angle-only measurement accepted
    summary: dict = field(default_factory=dict)


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _detection_sigma(confidence: float, length: float) -> float:
    angular = np.radians(DETECTION_SIGMA_DEG) / max(0.05, min(1.0, confidence))
    pixel = ENDPOINT_SIGMA_PX / max(length, 1.0)
    return float(np.hypot(angular, pixel))


def _median_smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    out = values.copy()
    for index in range(len(values)):
        window = values[max(0, index - radius): index + radius + 1]
        out[index] = np.median(window, axis=0)
    return out


def track_club(
    points: np.ndarray,
    valid: np.ndarray,
    confidence: np.ndarray,
    hands: np.ndarray,
    fps: float,
    measurements: dict[int, list[tuple[float, float, str]]] | None = None,
    reliable: float = 0.3,
) -> ClubTrack | None:
    """Track the club over a clip. Returns None without two accepted detections.

    `confidence` is the per-frame detection confidence (the weaker of the two
    landmarks). `hands` is the per-frame hands point (filled). `measurements`
    maps a frame to angle-only observations (angle grip-to-head, sigma radians, source).
    """
    n = len(points)
    measurements = measurements or {}
    grip = np.asarray(points[:, SHAFT], dtype=np.float64)
    head = np.asarray(points[:, HOSEL], dtype=np.float64)
    vector = head - grip
    length = np.linalg.norm(vector, axis=1)
    present = (valid[:, SHAFT] & valid[:, HOSEL] & np.isfinite(vector).all(axis=1) & (length > 1e-6))
    confidence = np.nan_to_num(np.asarray(confidence, dtype=np.float64))
    # Geometric swap: the grip is the end at the hands.
    hands = np.asarray(hands, dtype=np.float64)
    grip_gap = np.linalg.norm(grip - hands, axis=1)
    head_gap = np.linalg.norm(head - hands, axis=1)
    # When one end is clearly nearer the hands, geometry fixes the direction and
    # continuity may not reverse it: near impact the shaft turns so fast per frame
    # that the reversed direction can sit closer to the prediction.
    oriented = present & np.isfinite(hands).all(axis=1) & (np.abs(grip_gap - head_gap) >= ORIENTED_FRACTION * length)
    geometric_swap = oriented & (head_gap < grip_gap)
    angle = np.arctan2(vector[:, 1], vector[:, 0])
    angle = np.where(geometric_swap, angle + np.pi, angle)

    seeds = np.flatnonzero(oriented & (confidence >= reliable))
    if len(seeds) == 0:
        seeds = np.flatnonzero(present & (confidence >= reliable))
    if len(seeds) == 0:
        seeds = np.flatnonzero(present)
    if len(seeds) < 2:
        return None
    start = int(seeds[0])

    q = np.radians(ACCELERATION_SIGMA_DEG) * (30.0 / max(fps, 1.0)) ** 2
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = q * q * np.array([[0.25, 0.5], [0.5, 1.0]])
    predicted_x = np.zeros((n, 2))
    predicted_p = np.zeros((n, 2, 2))
    filtered_x = np.zeros((n, 2))
    filtered_p = np.zeros((n, 2, 2))
    accepted = np.zeros(n, dtype=bool)
    rejected = np.zeros(n, dtype=bool)
    swapped = geometric_swap.copy()
    measurement_used = np.zeros(n, dtype=bool)

    def update(x, p, observed, sigma):
        innovation = _wrap(observed - x[0])
        s = p[0, 0] + sigma * sigma
        gain = p[:, 0] / s
        x = x + gain * innovation
        p = p - np.outer(gain, p[0, :])
        return x, p

    def gate(x, p, observed, sigma):
        innovation = _wrap(observed - x[0])
        s = p[0, 0] + sigma * sigma
        return innovation * innovation <= GATE_SIGMAS ** 2 * s or abs(innovation) <= np.radians(ALWAYS_ACCEPT_DEG)

    x = np.array([angle[start], 0.0])
    p = np.diag([_detection_sigma(confidence[start], length[start]) ** 2, np.radians(10.0) ** 2])
    misses = 0
    for frame in range(start, n):
        if frame > start:
            x = transition @ x
            p = transition @ p @ transition.T + process
        predicted_x[frame], predicted_p[frame] = x, p
        if present[frame]:
            sigma = _detection_sigma(confidence[frame], length[frame])
            if gate(x, p, angle[frame], sigma):
                x, p = update(x, p, angle[frame], sigma)
                accepted[frame] = True
                misses = 0
            elif confidence[frame] >= REINIT_CONFIDENCE:
                # A confident detection outruns the motion model (fast downswing):
                # trust it and restart the state from it.
                x = np.array([angle[frame], x[1]])
                p = np.diag([sigma ** 2, np.radians(30.0) ** 2])
                accepted[frame] = True
                misses = 0
            elif not oriented[frame] and gate(x, p, angle[frame] + np.pi, sigma):
                angle[frame] += np.pi
                swapped[frame] = ~swapped[frame]
                x, p = update(x, p, angle[frame], sigma)
                accepted[frame] = True
                misses = 0
            else:
                rejected[frame] = True
                if confidence[frame] >= reliable:
                    misses += 1
                if misses >= REINIT_AFTER:
                    # Track lost: restart from this detection.
                    x = np.array([angle[frame], 0.0])
                    p = np.diag([sigma ** 2, np.radians(30.0) ** 2])
                    rejected[frame], accepted[frame] = False, True
                    misses = 0
        if not accepted[frame]:
            for observed, sigma, _source in measurements.get(frame, []):
                if gate(x, p, observed, sigma):
                    x, p = update(x, p, observed, sigma)
                    measurement_used[frame] = True
        filtered_x[frame], filtered_p[frame] = x, p

    # Rauch-Tung-Striebel smoother.
    smoothed_x = filtered_x.copy()
    smoothed_p = filtered_p.copy()
    for frame in range(n - 2, start - 1, -1):
        prior_p = predicted_p[frame + 1]
        gain = filtered_p[frame] @ transition.T @ np.linalg.inv(prior_p)
        difference = smoothed_x[frame + 1] - predicted_x[frame + 1]
        smoothed_x[frame] = filtered_x[frame] + gain @ difference
        smoothed_p[frame] = filtered_p[frame] + gain @ (smoothed_p[frame + 1] - prior_p) @ gain.T
    smoothed_x[:start] = smoothed_x[start]
    smoothed_p[:start] = smoothed_p[start]
    theta = smoothed_x[:, 0]
    sigma_deg = np.degrees(np.sqrt(np.maximum(smoothed_p[:, 0, 0], 0.0)))

    good = np.flatnonzero(accepted)
    frames = np.arange(n)
    # Projected length and grip offset from the hands, from accepted detections.
    corrected_grip = np.where(swapped[:, None], head, grip)
    corrected_head = np.where(swapped[:, None], grip, head)
    track_length = _median_smooth(np.interp(frames, good, length[good]))
    offsets = corrected_grip - hands
    usable = good[np.isfinite(offsets[good]).all(axis=1)]
    if len(usable):
        offset = np.column_stack([np.interp(frames, usable, offsets[usable, axis]) for axis in range(2)])
        offset = _median_smooth(offset)
    else:
        offset = np.zeros((n, 2))
    anchor = np.where(np.isfinite(hands).all(axis=1)[:, None], hands + offset,
                      np.column_stack([np.interp(frames, good, corrected_grip[good, axis]) for axis in range(2)]))
    direction = np.column_stack([np.cos(theta), np.sin(theta)])
    tracked = np.stack([anchor, anchor + track_length[:, None] * direction], axis=1)
    keep = accepted & (confidence >= reliable)
    tracked[keep, 0] = corrected_grip[keep]
    tracked[keep, 1] = corrected_head[keep]
    summary = {
        "method": "rigid_stick_kalman_rts",
        "detections": int(present.sum()),
        "accepted": int(accepted.sum()),
        "rejected": int(rejected.sum()),
        "swapped": int((swapped & accepted).sum()),
        "angle_measurements_used": int(measurement_used.sum()),
        "sigma_deg_median": round(float(np.median(sigma_deg)), 2),
        "sigma_deg_max": round(float(np.max(sigma_deg)), 2),
        "rejected_frames": [int(f) for f in np.flatnonzero(rejected)],
        "swapped_frames": [int(f) for f in np.flatnonzero(swapped & accepted)],
    }
    return ClubTrack(tracked, theta, sigma_deg, accepted, rejected, swapped & accepted, measurement_used, summary)


class ShaftPredictor:
    """Online shaft prediction for choosing a club crop during 2D detection.

    Keeps the last confident detections (grip, hosel). For a new frame it
    extrapolates the angle at the last angular speed, anchors the shaft at the
    current hands and returns a box around that predicted shaft, so the club
    model sees the club large even when the club detector found no box.
    """

    def __init__(self, reliable: float = 0.3, max_age: int = 6, margin: float = 0.3):
        self.reliable = reliable
        self.max_age = max_age
        self.margin = margin
        self.history: list[tuple[int, np.ndarray, np.ndarray]] = []

    def observe(self, frame: int, grip: np.ndarray, head: np.ndarray, confidence: float) -> None:
        if confidence < self.reliable or not (np.isfinite(grip).all() and np.isfinite(head).all()):
            return
        if np.linalg.norm(head - grip) < 1e-6:
            return
        self.history = (self.history + [(frame, np.asarray(grip, float), np.asarray(head, float))])[-3:]

    def predict_box(self, frame: int, hands: np.ndarray | None, size: tuple[int, int]) -> tuple[np.ndarray, dict] | None:
        if not self.history or frame - self.history[-1][0] > self.max_age:
            return None
        last_frame, grip, head = self.history[-1]
        vector = head - grip
        length = float(np.median([np.linalg.norm(h - g) for _, g, h in self.history]))
        theta = float(np.arctan2(vector[1], vector[0]))
        speed = 0.0
        if len(self.history) >= 2 and last_frame - self.history[-2][0] <= 2:
            before = self.history[-2][2] - self.history[-2][1]
            speed = _wrap(theta - float(np.arctan2(before[1], before[0]))) / (last_frame - self.history[-2][0])
            speed = float(np.clip(speed, -np.radians(60), np.radians(60)))
        steps = frame - last_frame
        predicted = theta + speed * steps
        anchor = np.asarray(hands, float) if hands is not None and np.isfinite(hands).all() else grip
        tip = anchor + length * np.array([np.cos(predicted), np.sin(predicted)])
        spread = self.margin * length + 0.5 * abs(speed * steps) * length
        corners = np.vstack([anchor, tip])
        low, high = corners.min(axis=0) - spread, corners.max(axis=0) + spread
        centre = (low + high) / 2
        half = np.maximum((high - low) / 2, 0.4 * length)
        width, height = size
        box = np.array([max(0.0, centre[0] - half[0]), max(0.0, centre[1] - half[1]),
                        min(float(width), centre[0] + half[0]), min(float(height), centre[1] + half[1])])
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return None
        return box, {"from_frame": int(last_frame), "predicted_angle_deg": round(float(np.degrees(predicted)), 1),
                     "length_px": round(length, 1), "box_xyxy": [round(float(v), 1) for v in box]}


def apply_track(points: np.ndarray, valid: np.ndarray, track: ClubTrack) -> tuple[np.ndarray, np.ndarray]:
    """Raw points with swapped detections exchanged and rejected detections marked invalid."""
    points = points.copy()
    valid = valid.copy()
    swapped = track.swapped
    points[swapped, SHAFT], points[swapped, HOSEL] = points[swapped, HOSEL].copy(), points[swapped, SHAFT].copy()
    valid[track.rejected, SHAFT] = False
    valid[track.rejected, HOSEL] = False
    points[track.rejected, SHAFT] = np.nan
    points[track.rejected, HOSEL] = np.nan
    return points, valid
