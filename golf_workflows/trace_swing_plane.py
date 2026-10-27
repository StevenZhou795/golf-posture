"""Draw a smoothed golf-clubhead path on a video from the first shaft to the last.

The club mask comes from the YOLO26n-seg weights trained for this project.
Wrists (or shoulders) from YOLO26n-pose decide which end of that mask is the
grip. The clubhead path is drawn as a smoothed curve from the start of the
swing through the finish.

Usage:
    python -m golf_workflows.trace_swing_plane
    python -m golf_workflows.trace_swing_plane path/to/swing.mp4
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict_pose import COCO_KEYPOINTS, ROOT
from predict_seg import MODEL_PATH, is_golf_club
from golf_workflows.swing_phases import (
    SwingPhases,
    WristFallback,
    detect_phases,
    phase_name,
    phase_progress,
    save_phase_chart,
)

OUTPUT_DIR = ROOT / "runs" / "swing_plane"
VIDEO_DIR = ROOT / "user_swings"
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}
POSE_MODEL = "yolo26n-pose.pt"
WRIST_INDEXES = (
    COCO_KEYPOINTS.index("left_wrist"),
    COCO_KEYPOINTS.index("right_wrist"),
)
SHOULDER_INDEXES = (
    COCO_KEYPOINTS.index("left_shoulder"),
    COCO_KEYPOINTS.index("right_shoulder"),
)
CLUB_CONFIDENCE = 0.25
MIN_SHAFT_PX = 12.0
MIN_THINNESS = 8.0
MAX_FRAME_GAP = 6
JUMP_FRACTION = 0.45
NEW_WEIGHT = 0.72
KEYPOINT_CONFIDENCE = 0.25
PATH_SAMPLES_PER_SPAN = 8
PATH_SMOOTH_PASSES = 2
HEAD_END_FRACTION = 0.25
MIN_HEAD_POINTS = 6
FACE_AXIS_TOLERANCE_DEG = 25.0


@dataclass
class ClubSample:
    frame: int
    grip: np.ndarray | None
    clubhead: np.ndarray
    confidence: float
    linked: bool
    face_from_shaft_deg: float | None = None
    face_from_horizontal_deg: float | None = None
    face_confidence: float | None = None
    face_axis: np.ndarray | None = None


def _pca_axes(points: np.ndarray):
    """Center, major unit axis, minor unit axis, and the two variances."""
    center = points.mean(axis=0)
    centered = points - center
    _eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    eigenvalues = np.asarray(_eigenvalues, dtype=np.float64)
    major_index = int(np.argmax(eigenvalues))
    minor_index = 1 - major_index
    major_axis = eigenvectors[:, major_index]
    minor_axis = eigenvectors[:, minor_index]
    return (
        center,
        major_axis,
        minor_axis,
        float(eigenvalues[major_index]),
        float(eigenvalues[minor_index]),
    )


def club_axis(polygon) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Major axis of a club contour: start, end, center, variance ratio, and minor axis."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 2:
        return None
    center, axis, minor_axis, major, minor = _pca_axes(points)
    if major < 1.0:
        return None
    centered = points - center
    projections = centered @ axis
    start = center + axis * float(projections.min())
    end = center + axis * float(projections.max())
    return start, end, center, major / max(minor, 1e-6), minor_axis


def _span(points: np.ndarray, axis: np.ndarray) -> float:
    projections = points @ axis
    return float(projections.max() - projections.min())


def _axis_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Unsigned angle between two undirected axes, from 0 to 90."""
    cosine = abs(float(np.dot(first, second)))
    cosine = min(1.0, max(0.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def _rect_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    _center, sides, angle = cv2.minAreaRect(points.astype(np.float32))
    if float(sides[0]) <= 0.0 and float(sides[1]) <= 0.0:
        return None
    radians = np.deg2rad(angle)
    along_width = np.array([np.cos(radians), np.sin(radians)], dtype=np.float64)
    across_width = np.array([-np.sin(radians), np.cos(radians)], dtype=np.float64)
    return along_width, across_width


def _signed_face_degrees(shaft: np.ndarray, face: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Degrees from square-to-shaft, degrees from horizontal, and a stable face axis."""
    perpendicular = np.array([-shaft[1], shaft[0]], dtype=np.float64)
    if float(np.dot(face, perpendicular)) < 0.0:
        face = -face
    cross = float(perpendicular[0] * face[1] - perpendicular[1] * face[0])
    from_shaft = float(np.degrees(np.arctan2(cross, float(np.dot(perpendicular, face)))))
    from_horizontal = float(np.degrees(np.arctan2(face[1], face[0])))
    if from_horizontal > 90.0:
        from_horizontal -= 180.0
    elif from_horizontal <= -90.0:
        from_horizontal += 180.0
    return from_shaft, from_horizontal, face


def clubhead_face(polygon, grip, clubhead) -> dict | None:
    """Heel-to-toe direction of the clubhead end of a mask. None if the head is too thin."""
    if grip is None or clubhead is None:
        return None
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    grip = np.asarray(grip, dtype=np.float64)
    clubhead = np.asarray(clubhead, dtype=np.float64)
    shaft = clubhead - grip
    length = float(np.linalg.norm(shaft))
    if length < MIN_SHAFT_PX or len(points) < MIN_HEAD_POINTS:
        return None
    shaft_axis = shaft / length
    along = (points - grip) @ shaft_axis
    head_points = points[along >= (1.0 - HEAD_END_FRACTION) * length]
    shaft_points = points[along < (1.0 - HEAD_END_FRACTION) * length]
    if len(head_points) < MIN_HEAD_POINTS or len(shaft_points) < 3:
        return None
    perpendicular = np.array([-shaft_axis[1], shaft_axis[0]])
    shaft_width = _span(shaft_points, perpendicular)
    if shaft_width <= 1.0:
        return None
    # The heel and toe are the outline points that sit outside the shaft.
    offset = np.abs((head_points - grip) @ perpendicular)
    bulge = head_points[offset > shaft_width * 0.6]
    if len(bulge) < MIN_HEAD_POINTS:
        return None
    _center, major_axis, minor_axis, _major_value, _minor_value = _pca_axes(bulge)
    # The face line is the bulge axis that lies across the shaft.
    if abs(float(np.dot(minor_axis, shaft_axis))) <= abs(float(np.dot(major_axis, shaft_axis))):
        face_axis = minor_axis
    else:
        face_axis = major_axis
    rect_axes = _rect_axes(bulge)
    if rect_axes is None:
        return None
    if min(_axis_degrees(face_axis, axis) for axis in rect_axes) > FACE_AXIS_TOLERANCE_DEG:
        return None
    head_width = _span(bulge, face_axis)
    if head_width <= shaft_width:
        return None
    from_shaft, from_horizontal, face_axis = _signed_face_degrees(shaft_axis, face_axis)
    return {
        "from_shaft_deg": from_shaft,
        "from_horizontal_deg": from_horizontal,
        "confidence": head_width / shaft_width,
        "axis": face_axis,
    }


def orient_shaft(start, end, anchor, previous_head):
    """Return (grip, clubhead). The grip is the end nearer the hands."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if anchor is not None:
        anchor = np.asarray(anchor, dtype=np.float64)
        if np.linalg.norm(start - anchor) <= np.linalg.norm(end - anchor):
            return start, end
        return end, start
    if previous_head is not None:
        previous_head = np.asarray(previous_head, dtype=np.float64)
        if np.linalg.norm(start - previous_head) < np.linalg.norm(end - previous_head):
            return end, start
        return start, end
    if start[1] >= end[1]:
        return end, start
    return start, end


def golfer_index(result) -> int | None:
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    if result.boxes is None or len(result.boxes) == 0:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()
    limit = min(len(result.keypoints), len(boxes))
    widths = np.maximum(0.0, boxes[:limit, 2] - boxes[:limit, 0])
    heights = np.maximum(0.0, boxes[:limit, 3] - boxes[:limit, 1])
    return int(np.argmax(widths * heights))


def _mean_keypoints(xy, confidence, indexes) -> np.ndarray | None:
    chosen = []
    for index in indexes:
        if confidence is not None and float(confidence[index]) < KEYPOINT_CONFIDENCE:
            continue
        point = xy[index]
        if float(point[0]) <= 0.0 and float(point[1]) <= 0.0:
            continue
        chosen.append(point)
    if not chosen:
        return None
    return np.mean(chosen, axis=0)


def body_anchor(result, index: int) -> np.ndarray | None:
    """A point in the hands, used to tell the grip from the clubhead."""
    keypoints = result.keypoints
    xy = keypoints.xy[index].cpu().numpy()
    confidence = None
    if keypoints.conf is not None:
        confidence = keypoints.conf[index].cpu().numpy()
    for indexes in (WRIST_INDEXES, SHOULDER_INDEXES):
        point = _mean_keypoints(xy, confidence, indexes)
        if point is not None:
            return np.asarray(point, dtype=np.float64)
    if result.boxes is not None and index < len(result.boxes):
        x1, y1, x2, y2 = result.boxes.xyxy[index].cpu().tolist()
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
    return None


def best_club(result, names, anchor, previous_head, diagonal):
    """Pick the golf-club mask and split it into grip and clubhead."""
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return None

    chosen = None
    polygons = result.masks.xy
    class_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    for index, class_id in enumerate(class_ids):
        class_id = int(class_id)
        if not is_golf_club(str(names[class_id])):
            continue
        if index >= len(polygons):
            continue
        axis = club_axis(polygons[index])
        if axis is None:
            continue
        start, end, center, thinness, _minor_axis = axis
        length = float(np.linalg.norm(end - start))
        if length < MIN_SHAFT_PX:
            continue
        grip, head = orient_shaft(start, end, anchor, previous_head)
        elongated = thinness >= MIN_THINNESS
        if not elongated:
            grip = None
            head = center
        face = clubhead_face(polygons[index], grip, head) if elongated else None
        reference = grip if grip is not None else head
        score = float(confidences[index]) + 0.2 * min(length / diagonal, 0.5)
        if not elongated:
            score -= 0.75
        if anchor is not None:
            score -= 0.4 * float(np.linalg.norm(reference - np.asarray(anchor)) / diagonal)
        if chosen is None or score > chosen[0]:
            chosen = (score, grip, head, float(confidences[index]), face)
    if chosen is None:
        return None
    _, grip, head, confidence, face = chosen
    return grip, head, confidence, face


def stabilize(grip, head, confidence, frame_index, previous, diagonal, face=None) -> ClubSample | None:
    """Drop a club that teleports, and ease the shaft so the plane does not jitter."""
    head = np.asarray(head, dtype=np.float64)
    if grip is not None:
        grip = np.asarray(grip, dtype=np.float64)
    linked = False
    if previous is not None:
        gap = frame_index - previous.frame
        jump = float(np.linalg.norm(head - previous.clubhead))
        if gap <= MAX_FRAME_GAP:
            if jump > JUMP_FRACTION * diagonal:
                return None
            linked = True
            if gap == 1:
                head = NEW_WEIGHT * head + (1.0 - NEW_WEIGHT) * previous.clubhead
                if grip is not None and previous.grip is not None:
                    grip = NEW_WEIGHT * grip + (1.0 - NEW_WEIGHT) * previous.grip
    return ClubSample(
        frame_index,
        None if grip is None else grip.copy(),
        head.copy(),
        confidence,
        linked,
        None if face is None else float(face["from_shaft_deg"]),
        None if face is None else float(face["from_horizontal_deg"]),
        None if face is None else float(face["confidence"]),
        None if face is None else np.asarray(face["axis"], dtype=np.float64).copy(),
    )


def phase_color_t(span: float) -> tuple[int, int, int]:
    """Blue at span 0, red at span 1."""
    span = min(1.0, max(0.0, float(span)))
    hue = int(round((1.0 - span) * 120))
    pixel = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(pixel[0]), int(pixel[1]), int(pixel[2])


def phase_color(index: int, count: int) -> tuple[int, int, int]:
    """Blue at the start of the swing, red at the latest sample."""
    span = 0.0 if count <= 1 else index / (count - 1)
    return phase_color_t(span)


def smooth_points(points: list[np.ndarray]) -> list[np.ndarray]:
    """Ease jitter along a clubhead run. Endpoints stay put."""
    current = [np.asarray(point, dtype=np.float64).copy() for point in points]
    if len(current) < 3:
        return current
    for _ in range(PATH_SMOOTH_PASSES):
        updated = [current[0].copy()]
        for index in range(1, len(current) - 1):
            updated.append(0.25 * current[index - 1] + 0.5 * current[index] + 0.25 * current[index + 1])
        updated.append(current[-1].copy())
        current = updated
    return current


def catmull_rom(p0, p1, p2, p3, t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def linear_span(p1, p2, t1: float, t2: float, sections: int):
    points = []
    times = []
    for step in range(sections):
        local = step / sections
        points.append((1.0 - local) * p1 + local * p2)
        times.append(t1 + (t2 - t1) * local)
    return points, times


def curve_span(p0, p1, p2, p3, t1: float, t2: float, sections: int):
    """Curve from p1 to p2. A straight span is used if the curve is unusable."""
    points = []
    times = []
    for step in range(sections):
        local = step / sections
        points.append(catmull_rom(p0, p1, p2, p3, local))
        times.append(t1 + (t2 - t1) * local)
    if not np.isfinite(np.stack(points)).all():
        return linear_span(p1, p2, t1, t2, sections)
    return points, times


@dataclass
class ClubheadRun:
    path: list[np.ndarray]
    path_times: list[float]


def clubhead_runs(samples: list[ClubSample], phases: SwingPhases | None = None) -> list[ClubheadRun]:
    """Linked clubhead runs, the smoothed path, and each step along that path."""
    count = len(samples)
    grouped: list[tuple[list[ClubSample], list[float]]] = []
    group: list[ClubSample] = []
    times: list[float] = []
    for index, sample in enumerate(samples):
        if group and not sample.linked:
            grouped.append((group, times))
            group = []
            times = []
        if phases is not None and phases.address is not None:
            span = phase_progress(sample.frame, phases)
        else:
            span = 0.0 if count <= 1 else index / (count - 1)
        group.append(sample)
        times.append(span)
    if group:
        grouped.append((group, times))

    runs = []
    for group, times in grouped:
        smoothed = smooth_points([sample.clubhead for sample in group])
        path: list[np.ndarray] = []
        path_times: list[float] = []
        if len(smoothed) == 1:
            runs.append(ClubheadRun(list(smoothed), list(times)))
            continue
        last = len(smoothed) - 1
        for index in range(last):
            p1 = smoothed[index]
            p2 = smoothed[index + 1]
            t1 = times[index]
            t2 = times[index + 1]
            if len(smoothed) < 3:
                span_points, span_times = linear_span(p1, p2, t1, t2, PATH_SAMPLES_PER_SPAN)
            else:
                p0 = smoothed[index - 1] if index > 0 else p1
                p3 = smoothed[index + 2] if index + 2 <= last else p2
                span_points, span_times = curve_span(p0, p1, p2, p3, t1, t2, PATH_SAMPLES_PER_SPAN)
            path.extend(span_points)
            path_times.extend(span_times)
        path.append(smoothed[-1])
        path_times.append(times[-1])
        runs.append(ClubheadRun(path, path_times))
    return runs


def as_point(point, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(min(width - 1, max(0.0, float(point[0]))))),
        int(round(min(height - 1, max(0.0, float(point[1]))))),
    )


def outlined_text(image, text: str, origin: tuple[int, int], scale: float) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(canvas) -> None:
    width = canvas.shape[1]
    scale = max(0.45, width / 1600.0)
    radius = max(5, int(round(7 * scale)))
    y = 24 + radius
    outlined_text(canvas, "clubhead path", (16, y), scale)
    y += int(round(28 * scale))
    start = phase_color(0, 2)
    finish = phase_color(1, 2)
    cv2.circle(canvas, (16 + radius, y), radius, start, -1, cv2.LINE_AA)
    outlined_text(canvas, "start", (16 + radius * 2 + 8, y + int(round(5 * scale))), scale)
    finish_x = 16 + int(round(150 * scale))
    cv2.circle(canvas, (finish_x, y), radius, finish, -1, cv2.LINE_AA)
    outlined_text(canvas, "finish", (finish_x + radius + 8, y + int(round(5 * scale))), scale)


def render_swing_plane(
    frame,
    samples: list[ClubSample],
    frame_index: int,
    phases: SwingPhases | None = None,
):
    """Paint the smoothed clubhead path collected so far."""
    canvas = frame.copy()
    if not samples:
        return canvas

    height, width = canvas.shape[:2]
    trace_px = max(2, width // 480)
    marker = max(4, width // 280)
    runs = clubhead_runs(samples, phases)
    label = phase_name(frame_index, phases) if phases is not None else ""
    if label:
        outlined_text(canvas, label, (16, height - 24), max(0.5, width / 1400.0))

    for run in runs:
        points, times = run.path, run.path_times
        for index in range(len(points) - 1):
            color = phase_color_t(times[index])
            p0 = as_point(points[index], width, height)
            p1 = as_point(points[index + 1], width, height)
            cv2.line(canvas, p0, p1, (0, 0, 0), trace_px + 2, cv2.LINE_AA)
            cv2.line(canvas, p0, p1, color, trace_px, cv2.LINE_AA)

    origin = as_point(samples[0].clubhead, width, height)
    cv2.circle(canvas, origin, marker + 3, phase_color(0, max(len(samples), 2)), 2, cv2.LINE_AA)

    latest = samples[-1]
    if latest.frame == frame_index:
        head = as_point(latest.clubhead, width, height)
        color = phase_color(len(samples) - 1, len(samples))
        cv2.circle(canvas, head, marker, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, head, marker, (255, 255, 255), 1, cv2.LINE_AA)

    draw_legend(canvas)
    return canvas


def point_list(point) -> list[float] | None:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def collect_videos(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def open_writer(path: Path, fps: float, frame) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"WMV2"),
        fps or 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {path}")
    return writer


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_video(argument: Path | None) -> Path:
    if argument is not None:
        video = argument.expanduser()
        if not video.is_file():
            raise SystemExit(f"Video not found: {video}")
        if video.suffix.lower() not in VIDEO_EXTS:
            raise SystemExit(f"Unsupported video type: {video.suffix}")
        return video
    if not VIDEO_DIR.is_dir():
        raise SystemExit(f"Video folder not found: {VIDEO_DIR}")
    videos = collect_videos(VIDEO_DIR)
    if not videos:
        raise SystemExit(f"No videos found in {VIDEO_DIR}")
    return random.choice(videos)


def wrist_height(result, person: int | None) -> float:
    if person is None or result.keypoints is None:
        return float("nan")
    xy = result.keypoints.xy[person].cpu().numpy()
    confidence = result.keypoints.conf
    chosen = []
    for index in WRIST_INDEXES:
        if confidence is not None and float(confidence[person][index]) < KEYPOINT_CONFIDENCE:
            continue
        if float(xy[index][0]) <= 0.0 and float(xy[index][1]) <= 0.0:
            continue
        chosen.append(float(xy[index][1]))
    if not chosen:
        return float("nan")
    return float(np.mean(chosen))


def nearest_sample(samples: list[ClubSample], frame: int | None, max_gap: int = 3) -> ClubSample | None:
    """Nearest club sample to a phase frame. Prefer one whose face was measured."""
    if frame is None or not samples:
        return None
    nearby = [sample for sample in samples if abs(sample.frame - frame) <= max_gap]
    if not nearby:
        return None
    measured = [sample for sample in nearby if sample.face_from_shaft_deg is not None]
    pool = measured or nearby
    return min(pool, key=lambda sample: abs(sample.frame - frame))


def face_record(sample: ClubSample | None) -> dict | None:
    if sample is None:
        return None
    def rounded(value: float | None) -> float | None:
        return None if value is None else round(float(value), 1)

    return {
        "frame": sample.frame,
        "clubhead": point_list(sample.clubhead),
        "from_shaft_deg": rounded(sample.face_from_shaft_deg),
        "from_horizontal_deg": rounded(sample.face_from_horizontal_deg),
        "confidence": rounded(sample.face_confidence),
    }


def draw_face_mark(image, sample: ClubSample, label: str) -> None:
    if sample.face_axis is None or sample.face_from_shaft_deg is None:
        return
    height, width = image.shape[:2]
    half = max(14, width // 40)
    center = sample.clubhead
    axis = sample.face_axis / max(float(np.linalg.norm(sample.face_axis)), 1e-6)
    start = as_point(center - axis * half, width, height)
    end = as_point(center + axis * half, width, height)
    cv2.line(image, start, end, (255, 255, 255), 2, cv2.LINE_AA)
    text = f"{label} {sample.face_from_shaft_deg:+.0f} deg"
    origin = as_point(center + np.array([8.0, -8.0]), width, height)
    outlined_text(image, text, origin, max(0.45, width / 1600.0))


def clubhead_track(samples: list[ClubSample], frames: int) -> np.ndarray:
    track = np.full((frames, 2), np.nan)
    for sample in samples:
        if 0 <= sample.frame < frames:
            track[sample.frame] = sample.clubhead
    return track


def trace_video(video: Path, use_mediapipe: bool = False) -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Trained golf-club weights not found: {MODEL_PATH}")

    pose_model = YOLO(POSE_MODEL)
    club_model = YOLO(str(MODEL_PATH))
    wrists = WristFallback() if use_mediapipe else None
    pose_results = pose_model.predict(
        source=str(video),
        imgsz=640,
        stream=True,
        verbose=False,
    )

    samples: list[ClubSample] = []
    wrist_y: list[float] = []
    save_dir = OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = save_dir / f"{video.stem}_swing_plane.avi"
    still_path = save_dir / f"{video.stem}_swing_plane.jpg"
    frame_count = 0
    try:
        for frame_index, pose_result in enumerate(pose_results):
            frame = pose_result.orig_img
            diagonal = float(np.hypot(frame.shape[0], frame.shape[1]))
            person = golfer_index(pose_result)
            anchor = body_anchor(pose_result, person) if person is not None else None
            if wrists is not None and wrists.enabled and person is not None:
                xy = pose_result.keypoints.xy[person].cpu().numpy()
                confidence = (
                    pose_result.keypoints.conf[person].cpu().numpy()
                    if pose_result.keypoints.conf is not None
                    else np.ones(len(xy))
                )
                fused_xy, fused_conf = wrists.fuse(frame, xy, confidence)
                hands = [
                    fused_xy[index]
                    for index in WRIST_INDEXES
                    if fused_conf[index] >= KEYPOINT_CONFIDENCE
                ]
                if hands:
                    anchor = np.mean(hands, axis=0)
            wrist_y.append(wrist_height(pose_result, person))

            previous = samples[-1] if samples else None
            previous_head = None if previous is None else previous.clubhead
            club_result = club_model.predict(
                frame,
                imgsz=640,
                conf=CLUB_CONFIDENCE,
                retina_masks=True,
                max_det=3,
                verbose=False,
            )[0]
            detection = best_club(
                club_result,
                club_model.names,
                anchor,
                previous_head,
                max(diagonal, 1.0),
            )
            if detection is not None:
                grip, head, confidence, face = detection
                sample = stabilize(
                    grip,
                    head,
                    confidence,
                    frame_index,
                    previous,
                    max(diagonal, 1.0),
                    face,
                )
                if sample is not None:
                    samples.append(sample)

            frame_count = frame_index + 1
            if frame_index % 30 == 0:
                print(f"frame {frame_index}, plane samples {len(samples)}", flush=True)
    finally:
        if wrists is not None:
            wrists.close()

    if frame_count == 0:
        raise SystemExit(f"No frames decoded from {video}")

    fps = float(getattr(pose_model.predictor.dataset, "fps", 30) or 30)
    phases = detect_phases(clubhead_track(samples, frame_count), np.asarray(wrist_y), fps)
    chart_path = save_dir / f"{video.stem}_phases.png"
    save_phase_chart(chart_path, clubhead_track(samples, frame_count)[:, 1], phases, fps)

    capture = cv2.VideoCapture(str(video))
    writer = None
    last_frame = None
    written = 0
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            so_far = [sample for sample in samples if sample.frame <= frame_index]
            labeled = render_swing_plane(frame, so_far, frame_index, phases)
            if writer is None:
                writer = open_writer(saved, fps, labeled)
            writer.write(labeled)
            last_frame = labeled
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if last_frame is None:
        raise SystemExit(f"No frames decoded from {video}")

    for phase_frame, label in ((phases.top, "P4"), (phases.impact, "P7")):
        marked = nearest_sample(samples, phase_frame)
        if marked is not None:
            draw_face_mark(last_frame, marked, label)
    cv2.imwrite(str(still_path), last_frame)
    summary_path = save_dir / f"{video.stem}_swing_plane.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": relative(video),
                "video": relative(saved),
                "still": relative(still_path),
                "phases_chart": relative(chart_path),
                "club_model": relative(MODEL_PATH),
                "pose_model": POSE_MODEL,
                "frames": frame_count,
                "frames_written": written,
                "phases": phases.as_dict(),
                "p4_face": face_record(nearest_sample(samples, phases.top)),
                "p7_face": face_record(nearest_sample(samples, phases.impact)),
                "mediapipe_wrists": use_mediapipe,
                "mediapipe_wrists_replaced": 0 if wrists is None else wrists.replaced,
                "samples": [
                    {
                        "frame": sample.frame,
                        "confidence": round(sample.confidence, 4),
                        "linked": sample.linked,
                        "grip": point_list(sample.grip),
                        "clubhead": point_list(sample.clubhead),
                        "face_from_shaft_deg": (
                            None
                            if sample.face_from_shaft_deg is None
                            else round(sample.face_from_shaft_deg, 1)
                        ),
                        "face_from_horizontal_deg": (
                            None
                            if sample.face_from_horizontal_deg is None
                            else round(sample.face_from_horizontal_deg, 1)
                        ),
                        "face_confidence": (
                            None if sample.face_confidence is None else round(sample.face_confidence, 2)
                        ),
                    }
                    for sample in samples
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Source: {video}")
    print(f"Frames: {frame_count}")
    print(f"Phases: {phases.as_dict()}")
    print(f"Swing-plane samples: {len(samples)}")
    print(f"Annotated video: {saved}")
    print(f"Still: {still_path}")
    print(f"Trace: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a smoothed clubhead path on a video.")
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Swing video. Defaults to a random file in user_swings.",
    )
    parser.add_argument(
        "--mediapipe-wrists",
        action="store_true",
        help="Fill YOLO wrists below 0.25 confidence from MediaPipe Pose.",
    )
    args = parser.parse_args()
    trace_video(resolve_video(args.video), use_mediapipe=args.mediapipe_wrists)


if __name__ == "__main__":
    main()
