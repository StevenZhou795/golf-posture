"""Swing phases from a clubhead path, with a wrist fallback.

The idea follows GolfPosePro (MediaPipe wrist trajectory, a short moving
average, then velocity): address is the start of motion, the top is the
point farthest from address, impact is the fast return to that spot, and
follow-through is when the club slows down again. The clubhead is the
signal when it exists. Wrist height is used only when the club is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_SAMPLES = 8
MOTION_FRACTION = 0.12
IMPACT_NEAR = 0.35
DECAY_FRACTION = 0.35
SUSTAIN_FRAMES = 3


@dataclass
class SwingPhases:
    address: int | None
    top: int | None
    impact: int | None
    follow_through: int | None
    signal: str

    def as_dict(self) -> dict:
        return {
            "address": self.address,
            "top": self.top,
            "impact": self.impact,
            "follow_through": self.follow_through,
            "signal": self.signal,
        }


def smooth_window(fps: float) -> int:
    """About 0.15 s, at least five frames, and always odd."""
    size = int(round(max(5.0, 0.15 * float(fps or 30.0))))
    if size % 2 == 0:
        size += 1
    return size


def smooth_series(values: np.ndarray, fps: float) -> np.ndarray:
    """Moving average. NaNs are filled first so the filter does not punch holes."""
    series = np.asarray(values, dtype=np.float64).copy()
    if series.ndim != 1 or len(series) < 3:
        return series
    good = np.isfinite(series)
    if good.sum() < 2:
        return series
    frames = np.arange(len(series))
    series = np.interp(frames, frames[good], series[good])
    window = min(smooth_window(fps), len(series) if len(series) % 2 else len(series) - 1)
    if window < 3:
        return series
    return uniform_filter1d(series, size=window, mode="nearest")


def smooth_keypoints(points: np.ndarray, fps: float) -> np.ndarray:
    """Smooth each joint independently. Shape is (frames, joints, 2)."""
    smoothed = np.asarray(points, dtype=np.float64).copy()
    for joint in range(smoothed.shape[1]):
        for axis in range(smoothed.shape[2]):
            smoothed[:, joint, axis] = smooth_series(smoothed[:, joint, axis], fps)
    return smoothed


def _sustain(mask: np.ndarray, count: int) -> int | None:
    run = 0
    for index, flag in enumerate(mask):
        run = run + 1 if flag else 0
        if run >= count:
            return index - count + 1
    return None


def _prepare_track(track: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return smoothed xy, speed in px/frame, and a finite-count check."""
    points = np.asarray(track, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return None
    finite = np.isfinite(points).all(axis=1)
    if int(finite.sum()) < MIN_SAMPLES:
        return None
    frames = np.arange(len(points))
    filled = np.column_stack(
        [
            np.interp(frames, frames[finite], points[finite, 0]),
            np.interp(frames, frames[finite], points[finite, 1]),
        ]
    )
    smoothed = np.column_stack([smooth_series(filled[:, 0], fps), smooth_series(filled[:, 1], fps)])
    velocity = np.gradient(smoothed, axis=0)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    return smoothed, speed, finite


def detect_phases(
    clubhead: np.ndarray | None,
    wrist_y: np.ndarray | None,
    fps: float,
) -> SwingPhases:
    """Label address, top, impact, and follow-through on one swing."""
    empty = SwingPhases(None, None, None, None, "none")
    prepared = None
    signal = "clubhead"
    if clubhead is not None:
        prepared = _prepare_track(clubhead, fps)
    if prepared is None and wrist_y is not None:
        wrist = np.asarray(wrist_y, dtype=np.float64)
        track = np.column_stack([np.zeros(len(wrist)), wrist])
        prepared = _prepare_track(track, fps)
        signal = "wrist"
    if prepared is None:
        return empty

    smoothed, speed, _finite = prepared
    peak = float(np.max(speed))
    if peak <= 1e-6:
        return empty
    motion = speed >= max(1.5, MOTION_FRACTION * peak)
    onset = _sustain(motion, SUSTAIN_FRAMES)
    if onset is None:
        return SwingPhases(None, None, None, None, signal)
    address = max(0, onset - 1)
    origin = smoothed[address]
    distance = np.linalg.norm(smoothed - origin, axis=1)
    farthest = float(np.max(distance[address:]))
    if farthest <= 1e-6:
        return SwingPhases(address, None, None, None, signal)

    # The club has to travel away from address before a fast return can be impact.
    raised_at = np.flatnonzero(np.arange(len(distance)) > address)
    raised_at = raised_at[distance[raised_at] >= 0.55 * farthest]
    raised = int(raised_at[0]) if len(raised_at) else address
    after_raise = np.arange(len(speed)) >= raised
    near = distance <= IMPACT_NEAR * farthest
    fast = speed >= 0.45 * peak
    candidates = np.flatnonzero(after_raise & near & fast)
    if len(candidates):
        impact = int(candidates[np.argmax(speed[candidates])])
    else:
        later = np.flatnonzero(after_raise)
        impact = int(later[np.argmax(speed[later])]) if len(later) else raised
    if impact <= address:
        impact = min(len(speed) - 1, address + 1)
    top_slice = distance[address:impact + 1]
    top = int(address + np.argmax(top_slice)) if len(top_slice) else address

    decay_at = None
    if impact + 1 < len(speed):
        quiet = speed < DECAY_FRACTION * float(speed[impact])
        quiet[: impact + 1] = False
        decay_at = _sustain(quiet, SUSTAIN_FRAMES)
    follow = decay_at if decay_at is not None else len(speed) - 1
    follow = max(follow, min(len(speed) - 1, impact + 1))
    return SwingPhases(address, top, impact, follow, signal)


def _finite_row(points: np.ndarray) -> bool:
    return bool(np.isfinite(points).all())


def _unit(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return None
    return vector / length


def _up_component(direction: np.ndarray) -> float:
    """Height above the ground. Camera Y points down, so up is -Y.

    The 3D picture draws the ground as its XY plane. A bone parallel to that
    plane has an up-component near zero.
    """
    return abs(float(direction[1]))


def _first_parallel(
    poses: np.ndarray,
    start: int,
    end: int,
    joint_a: int,
    joint_b: int,
    threshold: float,
) -> int | None:
    """First frame in (start, end) where the bone is near parallel to the ground."""
    stop = min(end, len(poses) - 1)
    for frame in range(max(0, start) + 1, stop):
        if not (_finite_row(poses[frame, joint_a]) and _finite_row(poses[frame, joint_b])):
            continue
        direction = _unit(poses[frame, joint_a], poses[frame, joint_b])
        if direction is not None and _up_component(direction) <= threshold:
            return frame
    return None


def _closest_horizontal(poses: np.ndarray, start: int, end: int, joint_a: int, joint_b: int) -> int | None:
    best = None
    best_tilt = float("inf")
    stop = min(end, len(poses) - 1)
    for frame in range(max(0, start), stop + 1):
        if not (_finite_row(poses[frame, joint_a]) and _finite_row(poses[frame, joint_b])):
            continue
        direction = _unit(poses[frame, joint_a], poses[frame, joint_b])
        if direction is None:
            continue
        tilt = abs(float(direction[1]))
        if tilt < best_tilt:
            best_tilt = tilt
            best = frame
    return best


def _lag_angle(poses: np.ndarray, frame: int, elbow: int, wrist: int, shaft: int, hosel: int) -> float | None:
    if not all(_finite_row(poses[frame, joint]) for joint in (elbow, wrist, shaft, hosel)):
        return None
    forearm = _unit(poses[frame, elbow], poses[frame, wrist])
    club = _unit(poses[frame, shaft], poses[frame, hosel])
    if forearm is None or club is None:
        return None
    cosine = min(1.0, max(-1.0, float(np.dot(forearm, club))))
    return float(np.degrees(np.arccos(cosine)))


def find_checkpoint_frames(
    poses: np.ndarray,
    address: int | None,
    top: int | None,
    impact: int | None,
    lead: str = "left",
) -> dict[str, int | None]:
    """P2 through P10 from a 3D skeleton. P1, P4, and P7 are the phases already found.

    Camera y points down, so a horizontal bone has a near-zero y component.
    """
    shaft, hosel = 17, 18
    head = 10
    if lead == "left":
        shoulder, wrist = 11, 13
    else:
        shoulder, wrist = 14, 16
    positions = {
        "p1": address,
        "p2": None,
        "p3": None,
        "p4": top,
        "p5": None,
        "p6": None,
        "p7": impact,
        "p8": None,
        "p9": None,
        "p10": None,
    }
    if address is None or top is None or impact is None:
        return positions
    if not (0 <= address <= top <= impact < len(poses)):
        return positions

    # Halfway back is the first time the lead arm becomes parallel, before the top.
    positions["p3"] = _first_parallel(poses, address, top, shoulder, wrist, 0.25)
    # Takeaway is the shaft closest to the 3D view's ground, after address and before P3.
    p3 = positions["p3"]
    positions["p2"] = (
        _closest_horizontal(poses, address + 1, p3 - 1, shaft, hosel) if p3 is not None else None
    )
    hip = 4 if lead == "left" else 1
    hip_height = float(poses[address, hip, 1]) if _finite_row(poses[address, hip]) else None
    p5 = None
    for frame in range(top + 1, impact):
        if not (_finite_row(poses[frame, shaft]) and _finite_row(poses[frame, hosel])):
            continue
        direction = _unit(poses[frame, shaft], poses[frame, hosel])
        if direction is None or _up_component(direction) > 0.35:
            continue
        if hip_height is not None and _finite_row(poses[frame, wrist]):
            if float(poses[frame, wrist, 1]) >= hip_height:
                continue
        p5 = frame
        break
    positions["p5"] = p5
    # Pre-impact is the shaft closest to the 3D view's ground, after P5 and before the ball.
    positions["p6"] = (
        _closest_horizontal(poses, p5 + 1, impact - 1, shaft, hosel) if p5 is not None else None
    )

    last = len(poses) - 1
    positions["p9"] = _closest_horizontal(poses, impact + 1, last, shoulder, wrist)
    p9 = positions["p9"]
    # Release is the shaft closest to the 3D view's ground, after the ball and before P9.
    positions["p8"] = (
        _closest_horizontal(poses, impact + 1, p9 - 1, shaft, hosel) if p9 is not None else None
    )
    if p9 is None:
        p9 = impact
    nearest = None
    nearest_distance = float("inf")
    for frame in range(p9 + 1, last + 1):
        if not (_finite_row(poses[frame, wrist]) and _finite_row(poses[frame, head])):
            continue
        if not _finite_row(poses[frame, shoulder]):
            continue
        lead_side = poses[frame, wrist] - poses[frame, head]
        toward_lead = poses[frame, shoulder] - poses[frame, head]
        if float(np.dot(lead_side, toward_lead)) <= 0.0:
            continue
        distance = float(np.linalg.norm(poses[frame, wrist] - poses[frame, head]))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = frame
    positions["p10"] = nearest
    return positions


def wrist_shaft_degrees(poses: np.ndarray, frame: int | None, lead: str = "left") -> float | None:
    """Angle between the lead forearm and the shaft. This is a picture angle, not a sensor reading."""
    if frame is None or not (0 <= frame < len(poses)):
        return None
    elbow, wrist = (12, 13) if lead == "left" else (15, 16)
    return _lag_angle(poses, frame, elbow, wrist, 17, 18)


def phase_progress(frame: int, phases: SwingPhases) -> float:
    """0 at address, about 0.35 at the top, 0.7 at impact, 1 at follow-through."""
    marks = [
        (phases.address, 0.0),
        (phases.top, 0.35),
        (phases.impact, 0.70),
        (phases.follow_through, 1.0),
    ]
    marks = [(int(frame_index), value) for frame_index, value in marks if frame_index is not None]
    if not marks:
        return 0.0
    if frame <= marks[0][0]:
        return marks[0][1]
    if frame >= marks[-1][0]:
        return marks[-1][1]
    for (left, left_value), (right, right_value) in zip(marks, marks[1:]):
        if left <= frame <= right:
            span = max(right - left, 1)
            return left_value + (right_value - left_value) * ((frame - left) / span)
    return marks[-1][1]


def phase_name(frame: int, phases: SwingPhases) -> str:
    if phases.address is None:
        return ""
    if frame < phases.address:
        return "setup"
    if frame == phases.address:
        return "address"
    if phases.top is not None and frame < phases.top:
        return "backswing"
    if frame == phases.top:
        return "top"
    if phases.impact is not None and frame < phases.impact:
        return "downswing"
    if frame == phases.impact:
        return "impact"
    if phases.follow_through is not None and frame < phases.follow_through:
        return "follow-through"
    if frame == phases.follow_through:
        return "follow-through"
    return "finish"


def save_phase_chart(path, clubhead_y: np.ndarray, phases: SwingPhases, fps: float) -> None:
    """Wrist-style chart: smoothed height and speed, with phase lines."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    prepared = _prepare_track(np.column_stack([np.zeros(len(clubhead_y)), clubhead_y]), fps)
    if prepared is None:
        return
    _smoothed, speed, _finite = prepared
    figure = Figure(figsize=(8, 3.2), dpi=100)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    frames = np.arange(len(clubhead_y))
    axis.plot(frames, smooth_series(clubhead_y, fps), color="#1f77b4", label="clubhead y")
    twin = axis.twinx()
    twin.plot(frames, speed, color="#ff7f0e", alpha=0.8, label="speed")
    colors = {"address": "#2ca02c", "top": "#9467bd", "impact": "#d62728", "follow_through": "#7f7f7f"}
    for name, color in colors.items():
        frame_index = getattr(phases, name)
        if frame_index is not None:
            axis.axvline(frame_index, color=color, linestyle="--", label=name.replace("_", " "))
    axis.set_xlabel("frame")
    axis.set_ylabel("y (px, down)")
    twin.set_ylabel("speed (px/frame)")
    axis.legend(loc="upper left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)


class WristFallback:
    """Fill weak YOLO wrists from MediaPipe Pose. Off unless the task model loads."""

    LEFT = 9
    RIGHT = 10
    MP_LEFT = 15
    MP_RIGHT = 16
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )

    def __init__(self, model_path: str | None = None) -> None:
        self.landmarker = None
        self.replaced = 0
        self._timestamp_ms = 0
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core import base_options as base_options_lib
            from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
        except ImportError:
            print("mediapipe is not installed; wrist fallback is off", flush=True)
            return
        path = model_path or _pose_landmarker_path()
        if not os_path_exists(path):
            print(f"MediaPipe pose model not found: {path}", flush=True)
            return
        options = PoseLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=path),
            running_mode=running_mode_lib.VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._mp = mp

    @property
    def enabled(self) -> bool:
        return self.landmarker is not None

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def fuse(self, frame_bgr, xy: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace COCO wrists under 0.25 confidence when MediaPipe is confident."""
        xy = np.asarray(xy, dtype=np.float64).copy()
        confidence = np.asarray(confidence, dtype=np.float64).copy()
        if self.landmarker is None:
            return xy, confidence
        if confidence[self.LEFT] >= 0.25 and confidence[self.RIGHT] >= 0.25:
            return xy, confidence
        import cv2

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.pose_landmarks:
            return xy, confidence
        height, width = frame_bgr.shape[:2]
        landmarks = result.pose_landmarks[0]
        for coco_index, mp_index in ((self.LEFT, self.MP_LEFT), (self.RIGHT, self.MP_RIGHT)):
            if confidence[coco_index] >= 0.25:
                continue
            landmark = landmarks[mp_index]
            visibility = float(getattr(landmark, "visibility", 0.0) or 0.0)
            if visibility < 0.5:
                continue
            xy[coco_index] = (landmark.x * width, landmark.y * height)
            confidence[coco_index] = visibility
            self.replaced += 1
        return xy, confidence


def os_path_exists(path: str) -> bool:
    from pathlib import Path

    return Path(path).is_file()


def _pose_landmarker_path() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task")
