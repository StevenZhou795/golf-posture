"""BlazePose GHUM inference on video: timestamped frames, golfer selection, ROI second pass."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from blazepose_workflow.landmarks import J, NAMES, subset
from blazepose_workflow.models import model_path

POSES_PER_FRAME = 3
VISIBLE = 0.5
# The golfer is the candidate whose hip centre stays within this many body
# heights of the last one (hip slide in a swing is ~0.2); a farther candidate is someone else.
MAX_JUMP_BODY_HEIGHTS = 0.5
# After this long without the golfer, pick afresh (camera cut or pan).
ANCHOR_TIMEOUT_MS = 500
ROI_MARGIN = 0.15
ROI_MARGIN_TOP = 0.25
# A golfer ROI that covers this much of the frame gains nothing from cropping.
ROI_MAX_FRAME_FRACTION = 0.8
MIN_LONG_SIDE = 960


@dataclass
class PoseTrack:
    """Per-frame golfer pose. NaN where no golfer was found.

    image: (n, 23, 5) x, y in pixels, z (BlazePose relative depth, pixel scale),
        visibility, presence.
    world: (n, 23, 5) x, y, z in metres around the hip centre (GHUM), visibility,
        presence. Axes follow the camera: x right, y down, z away from the camera
        as MediaPipe returns it.
    """
    image: np.ndarray
    world: np.ndarray
    times_ms: np.ndarray
    found: np.ndarray
    candidates: np.ndarray
    size: tuple[int, int]
    fps: float
    source: list[str] = field(default_factory=list)
    crop: list[int] | None = None

    @property
    def frames(self) -> int:
        return len(self.times_ms)


def monotonic_ms(times: list[float]) -> list[int]:
    """Whole-millisecond timestamps, bumped by 1 ms where they do not increase.

    MediaPipe VIDEO mode rejects a timestamp that is not strictly larger than
    the previous one; containers sometimes repeat or round timestamps.
    """
    result: list[int] = []
    for value in times:
        result.append(next_stamp(result[-1] if result else None, value))
    return result


def next_stamp(previous: int | None, value: float) -> int:
    stamp = int(round(value))
    return previous + 1 if previous is not None and stamp <= previous else stamp


def body_box(pose: np.ndarray) -> np.ndarray | None:
    """x0, y0, x1, y1 of the visible landmarks of one (23, 5) pose."""
    seen = pose[pose[:, 3] >= VISIBLE, :2]
    if len(seen) < 4:
        return None
    return np.r_[seen.min(axis=0), seen.max(axis=0)]


def pose_center(pose: np.ndarray) -> np.ndarray:
    return (pose[J["left_hip"], :2] + pose[J["right_hip"], :2]) / 2


def select_golfer(candidates: list[np.ndarray], anchor: np.ndarray | None, extent: float | None,
                  size: tuple[int, int], prior_box: np.ndarray | None = None) -> int | None:
    """Index of the golfer among this frame's candidate poses (pixel coordinates).

    With an anchor (the golfer's last hip centre and body height), the nearest
    candidate within MAX_JUMP_BODY_HEIGHTS; otherwise the largest person whose
    hips are inside `prior_box`, or the largest and most central person.
    """
    if not candidates:
        return None
    centres = [pose_center(pose) for pose in candidates]
    if anchor is not None and extent:
        distances = [float(np.linalg.norm(centre - anchor)) for centre in centres]
        best = int(np.argmin(distances))
        return best if distances[best] <= MAX_JUMP_BODY_HEIGHTS * extent else None
    width, height = size
    scores = []
    for pose, centre in zip(candidates, centres):
        box = body_box(pose)
        if box is None:
            scores.append(-1.0)
            continue
        if prior_box is not None and not (prior_box[0] <= centre[0] <= prior_box[2]
                                          and prior_box[1] <= centre[1] <= prior_box[3]):
            scores.append(-1.0)
            continue
        body = (box[3] - box[1]) / height
        central = 1.0 - min(1.0, abs(centre[0] - width / 2) / (width / 2))
        scores.append(body * (0.5 + 0.5 * central))
    best = int(np.argmax(scores))
    return best if scores[best] >= 0 else None


def _landmarker(kind: str):
    from mediapipe.tasks.python import BaseOptions, vision
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path(kind))),
        running_mode=vision.RunningMode.VIDEO, num_poses=POSES_PER_FRAME)
    return vision.PoseLandmarker.create_from_options(options)


def run_pass(video: Path, kind: str, crop: list[int] | None = None,
             prior_box: np.ndarray | None = None, label: str = "pass") -> PoseTrack:
    """One BlazePose pass over the video, optionally on a fixed crop upscaled to MIN_LONG_SIDE."""
    import mediapipe as mp
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frames: list[int] = []
    images, worlds, found, counts = [], [], [], []
    anchor, extent, last_seen = None, None, None
    size = (0, 0)
    with _landmarker(kind) as landmarker:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stamp = next_stamp(frames[-1] if frames else None, float(capture.get(cv2.CAP_PROP_POS_MSEC)))
            frames.append(stamp)
            height, width = frame.shape[:2]
            size = (width, height)
            x0, y0, x1, y1 = crop if crop is not None else (0, 0, width, height)
            view = frame[y0:y1, x0:x1]
            scale = max(1.0, MIN_LONG_SIDE / max(view.shape[:2])) if crop is not None else 1.0
            if scale > 1.0:
                view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB)))
            result = landmarker.detect_for_video(image, stamp)
            crop_w, crop_h = x1 - x0, y1 - y0
            candidates, world_candidates = [], []
            for pose, world in zip(result.pose_landmarks, result.pose_world_landmarks):
                points = subset(pose)
                points[:, 0] = x0 + points[:, 0] * crop_w
                points[:, 1] = y0 + points[:, 1] * crop_h
                points[:, 2] = points[:, 2] * crop_w
                candidates.append(points)
                world_candidates.append(subset(world))
            if last_seen is not None and stamp - last_seen > ANCHOR_TIMEOUT_MS:
                anchor, extent = None, None
            chosen = select_golfer(candidates, anchor, extent, size, prior_box)
            counts.append(len(candidates))
            if chosen is None:
                images.append(np.full((len(NAMES), 5), np.nan))
                worlds.append(np.full((len(NAMES), 5), np.nan))
                found.append(False)
            else:
                pose = candidates[chosen]
                images.append(pose)
                worlds.append(world_candidates[chosen])
                found.append(True)
                box = body_box(pose)
                anchor = pose_center(pose)
                if box is not None:
                    extent = float(box[3] - box[1])
                last_seen = stamp
            if index % 50 == 0:
                print(f"  BlazePose {label} frame {index}: {len(candidates)} pose(s)", flush=True)
            index += 1
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video}")
    return PoseTrack(image=np.stack(images), world=np.stack(worlds), times_ms=np.asarray(frames, float),
                     found=np.asarray(found), candidates=np.asarray(counts), size=size, fps=fps,
                     source=[label if f else "none" for f in found], crop=crop)


def golfer_roi(track: PoseTrack) -> list[int] | None:
    """Fixed crop around every pose of the golfer, or None when it would cover most of the frame."""
    boxes = [body_box(pose) for pose, seen in zip(track.image, track.found) if seen]
    boxes = np.array([box for box in boxes if box is not None])
    if not len(boxes):
        return None
    x0, y0 = np.percentile(boxes[:, 0], 2), np.percentile(boxes[:, 1], 2)
    x1, y1 = np.percentile(boxes[:, 2], 98), np.percentile(boxes[:, 3], 98)
    w, h = x1 - x0, y1 - y0
    width, height = track.size
    roi = [int(max(0, x0 - ROI_MARGIN * w)), int(max(0, y0 - ROI_MARGIN_TOP * h)),
           int(min(width, x1 + ROI_MARGIN * w)), int(min(height, y1 + ROI_MARGIN * h))]
    area = (roi[2] - roi[0]) * (roi[3] - roi[1])
    if area >= ROI_MAX_FRAME_FRACTION * width * height or roi[2] - roi[0] < 32 or roi[3] - roi[1] < 32:
        return None
    return roi


def detect(video: Path, kind: str = "heavy", golfer_box: list[float] | None = None,
           crop: bool = True) -> tuple[PoseTrack, dict]:
    """Golfer pose for every frame: a full-frame pass, then a pass on the golfer ROI.

    Frames the ROI pass misses keep the full-frame result.
    """
    prior = None if golfer_box is None else np.asarray(golfer_box, float)
    first = run_pass(video, kind, prior_box=prior, label="full-frame")
    report = {"model": f"pose_landmarker_{kind}", "poses_per_frame": POSES_PER_FRAME,
              "full_frame_found": int(first.found.sum()), "frames": first.frames}
    roi = golfer_roi(first) if crop else None
    report["roi_xyxy"] = roi
    if roi is None:
        report["roi_pass"] = "skipped" if crop else "disabled"
        return first, report
    second = run_pass(video, kind, crop=roi, prior_box=np.asarray(roi, float), label="roi")
    missing = ~second.found & first.found
    second.image[missing] = first.image[missing]
    second.world[missing] = first.world[missing]
    second.found = second.found | first.found
    second.source = ["roi" if s == "roi" else ("full-frame" if m else "none")
                     for s, m in zip(second.source, missing)]
    report.update(roi_pass="applied", roi_found=int((np.array(second.source) == "roi").sum()),
                  filled_from_full_frame=int(missing.sum()))
    return second, report
