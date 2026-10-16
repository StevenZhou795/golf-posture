"""Lift golf swings to 3D with GolfPose-3D, using Ultralytics for the 2D input.

2D comes from the same models as trace_swing_plane.py: YOLO26n-pose for the
golfer and the trained YOLO26n-seg weights for the club. COCO joints are
converted to the Human3.6M order GolfPose expects, the club's grip end becomes
"shaft" and its clubhead end becomes "hosel", and the 17+2 MixSTE2 checkpoint
from https://github.com/MingHanLee/GolfPose lifts each swing to 3D.

Usage:
    python golf_pose_3d.py
    python golf_pose_3d.py path/to/swing.mp4
    python golf_pose_3d.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from ultralytics import YOLO, YOLOE

from predict_pose import ROOT
from predict_seg import MODEL_PATH
from swing_phases import WristFallback, detect_phases, find_checkpoint_frames, smooth_keypoints, wrist_shaft_degrees
from trace_swing_plane import (
    CLUB_CONFIDENCE,
    KEYPOINT_CONFIDENCE,
    POSE_MODEL,
    ClubSample,
    as_point,
    best_club,
    body_anchor,
    collect_videos,
    golfer_index,
    phase_color_t,
    outlined_text,
    VIDEO_DIR,
    open_writer,
    render_swing_plane,
    resolve_video,
    stabilize,
    wrist_height,
)

GOLFPOSE_DIR = ROOT / "third_party" / "golfpose"
CHECKPOINT = ROOT / "golfpose_checkpoints" / "golfpose_17+2_33.6_30.5_59.5.bin"
OUTPUT_DIR = ROOT / "runs" / "golf_pose_3d"

# Order from configs/mmpose/_base_/datasets/golfswing_golfer.py, first 17+2 points.
GOLFPOSE_JOINTS = [
    "root",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "shaft",
    "hosel",
]
JOINT = {name: index for index, name in enumerate(GOLFPOSE_JOINTS)}
BODY_JOINTS = 17
JOINTS_LEFT = [4, 5, 6, 11, 12, 13]
JOINTS_RIGHT = [1, 2, 3, 14, 15, 16]
BODY_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
BONES = [
    ("root", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("root", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("root", "spine"),
    ("spine", "thorax"),
    ("thorax", "neck_base"),
    ("neck_base", "head"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("shaft", "hosel"),
]

# COCO index for each GolfPose joint that maps one to one.
DIRECT_FROM_COCO = {
    "right_hip": 12,
    "right_knee": 14,
    "right_foot": 16,
    "left_hip": 11,
    "left_knee": 13,
    "left_foot": 15,
    "left_shoulder": 5,
    "left_elbow": 7,
    "left_wrist": 9,
    "right_shoulder": 6,
    "right_elbow": 8,
    "right_wrist": 10,
}
COCO_NOSE = 0
COCO_FACE = (3, 4, 1, 2)
NECK_TOWARD_NOSE = 0.5
HEAD_BEYOND_FACE = 0.25

RECEPTIVE_FIELD = 243
EMBED_DIM = 512
DEPTH = 8
NUM_HEADS = 8
MLP_RATIO = 2.0
CANVAS_W = 1280
CANVAS_H = 720
GOLFER_FRACTION = 0.7

BODY_RGB = {"left": (0, 255, 0), "right": (255, 128, 0), "center": (51, 153, 255), "club": (255, 255, 255)}
VIEW_ELEV = 15.0
VIEW_AZIM = 70.0
PANEL_ZOOM = 1.2
BALL_MODEL = ROOT / "yoloe-26n-seg.pt"
BALL_CONFIDENCE = 0.15
BALL_AREA_FRACTION = 0.02
POSITION_NAMES = {
    "p1": "address",
    "p2": "takeaway",
    "p3": "halfway back",
    "p4": "top",
    "p5": "early downswing",
    "p6": "pre-impact",
    "p7": "impact",
    "p8": "release",
    "p9": "finish",
    "p10": "end",
}


@dataclass
class Swing2D:
    frames: int
    fps: float
    size: tuple[int, int]
    points: np.ndarray
    valid: np.ndarray
    filled: np.ndarray
    boxes: np.ndarray
    samples: list[ClubSample]
    wrist_y: np.ndarray


def bone_side(a: str, b: str) -> str:
    if "shaft" in (a, b):
        return "club"
    names = a + b
    if "left" in names:
        return "left"
    if "right" in names:
        return "right"
    return "center"


def coco_to_golfpose(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """17 COCO joints to the 17 GolfPose body joints, with a validity flag per joint."""
    good = conf >= KEYPOINT_CONFIDENCE
    points = np.full((BODY_JOINTS, 2), np.nan)
    valid = np.zeros(BODY_JOINTS, dtype=bool)

    for name, coco_index in DIRECT_FROM_COCO.items():
        if good[coco_index]:
            points[JOINT[name]] = xy[coco_index]
            valid[JOINT[name]] = True

    def mean_of(*names: str) -> np.ndarray | None:
        if all(valid[JOINT[name]] for name in names):
            return np.mean([points[JOINT[name]] for name in names], axis=0)
        return None

    def put(name: str, value: np.ndarray | None) -> None:
        if value is not None:
            points[JOINT[name]] = value
            valid[JOINT[name]] = True

    root = mean_of("left_hip", "right_hip")
    thorax = mean_of("left_shoulder", "right_shoulder")
    put("root", root)
    put("thorax", thorax)
    if root is not None and thorax is not None:
        put("spine", (root + thorax) / 2.0)
    if thorax is not None:
        if good[COCO_NOSE]:
            put("neck_base", thorax + NECK_TOWARD_NOSE * (xy[COCO_NOSE] - thorax))
        face = [xy[index] for index in COCO_FACE if good[index]]
        if not face and good[COCO_NOSE]:
            face = [xy[COCO_NOSE]]
        if face:
            center = np.mean(face, axis=0)
            put("head", center + HEAD_BEYOND_FACE * (center - thorax))
    return points, valid


def fill_gaps(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear fill over time per joint. The first and last good values are held at the ends."""
    filled = points.copy()
    frames = np.arange(points.shape[0])
    for joint in range(points.shape[1]):
        good = valid[:, joint]
        if not good.any():
            continue
        for axis in range(2):
            filled[:, joint, axis] = np.interp(frames, frames[good], points[good, joint, axis])

    for joint in range(BODY_JOINTS):
        parent = BODY_PARENTS[joint]
        if not valid[:, joint].any() and parent >= 0:
            filled[:, joint] = filled[:, parent]
    wrists = (filled[:, JOINT["left_wrist"]] + filled[:, JOINT["right_wrist"]]) / 2.0
    if not valid[:, JOINT["shaft"]].any():
        filled[:, JOINT["shaft"]] = wrists
    if not valid[:, JOINT["hosel"]].any():
        filled[:, JOINT["hosel"]] = filled[:, JOINT["shaft"]]
    return filled


def detect_2d(video: Path, pose_model: YOLO, club_model: YOLO, wrists: WristFallback | None = None) -> Swing2D:
    pose_results = pose_model.predict(source=str(video), imgsz=640, stream=True, verbose=False)
    rows: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    samples: list[ClubSample] = []
    swing_wrist_y: list[float] = []
    size = (0, 0)
    for frame_index, pose_result in enumerate(pose_results):
        frame = pose_result.orig_img
        size = (frame.shape[1], frame.shape[0])
        diagonal = max(float(np.hypot(frame.shape[0], frame.shape[1])), 1.0)
        points = np.full((len(GOLFPOSE_JOINTS), 2), np.nan)
        valid = np.zeros(len(GOLFPOSE_JOINTS), dtype=bool)
        box = np.full(4, np.nan)

        anchor = None
        person = golfer_index(pose_result)
        if person is not None:
            anchor = body_anchor(pose_result, person)
            xy = pose_result.keypoints.xy[person].cpu().numpy().astype(np.float64)
            if pose_result.keypoints.conf is not None:
                conf = pose_result.keypoints.conf[person].cpu().numpy()
            else:
                conf = np.ones(len(xy))
            if wrists is not None and wrists.enabled:
                xy, conf = wrists.fuse(frame, xy, conf)
                hands = [xy[index] for index in (9, 10) if conf[index] >= KEYPOINT_CONFIDENCE]
                if hands:
                    anchor = np.mean(hands, axis=0)
            points[:BODY_JOINTS], valid[:BODY_JOINTS] = coco_to_golfpose(xy, conf)
            if pose_result.boxes is not None and person < len(pose_result.boxes):
                box = pose_result.boxes.xyxy[person].cpu().numpy().astype(np.float64)

        previous = samples[-1] if samples else None
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
            None if previous is None else previous.clubhead,
            diagonal,
        )
        if detection is not None:
            grip, head, confidence, face = detection
            sample = stabilize(grip, head, confidence, frame_index, previous, diagonal, face)
            if sample is not None:
                samples.append(sample)
                points[JOINT["hosel"]] = sample.clubhead
                valid[JOINT["hosel"]] = True
                if sample.grip is not None:
                    points[JOINT["shaft"]] = sample.grip
                    valid[JOINT["shaft"]] = True

        rows.append(points)
        flags.append(valid)
        boxes.append(box)
        swing_wrist_y.append(wrist_height(pose_result, person))
        if frame_index % 30 == 0:
            print(f"  2D frame {frame_index}, club samples {len(samples)}", flush=True)

    if not rows:
        raise SystemExit(f"No frames decoded from {video}")
    points = np.stack(rows)
    valid = np.stack(flags)
    if not valid[:, JOINT["root"]].any():
        raise SystemExit(f"No golfer with both hips found in {video}")
    fps = float(getattr(pose_model.predictor.dataset, "fps", 30) or 30)
    return Swing2D(
        frames=len(rows),
        fps=fps,
        size=size,
        points=points,
        valid=valid,
        filled=fill_gaps(points, valid),
        boxes=np.stack(boxes),
        samples=samples,
        wrist_y=np.asarray(swing_wrist_y, dtype=np.float64),
    )


def golf_canvas(swing: Swing2D) -> tuple[np.ndarray, dict]:
    """Place the swing on a 1280x720 canvas like the GolfPose cameras, then normalize."""
    boxes = swing.boxes[np.isfinite(swing.boxes).all(axis=1)]
    if len(boxes):
        height = float(np.median(boxes[:, 3] - boxes[:, 1]))
        center = np.array(
            [np.median((boxes[:, 0] + boxes[:, 2]) / 2.0), np.median((boxes[:, 1] + boxes[:, 3]) / 2.0)]
        )
    else:
        body = swing.filled[:, :BODY_JOINTS]
        height = float(np.median(body[..., 1].max(axis=1) - body[..., 1].min(axis=1)))
        center = np.median(body.reshape(-1, 2), axis=0)
    scale = CANVAS_H * GOLFER_FRACTION / max(height, 1.0)
    canvas = (swing.filled - center) * scale + np.array([CANVAS_W / 2.0, CANVAS_H / 2.0])
    normalized = canvas / CANVAS_W * 2.0 - np.array([1.0, CANVAS_H / CANVAS_W])
    params = {
        "canvas": [CANVAS_W, CANVAS_H],
        "golfer_fraction": GOLFER_FRACTION,
        "golfer_height_px": round(height, 2),
        "center_px": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "scale": round(float(scale), 5),
    }
    return normalized.astype(np.float32), params


def load_lifter(checkpoint: Path) -> torch.nn.Module:
    if not checkpoint.is_file():
        raise SystemExit(
            f"GolfPose checkpoint not found: {checkpoint}\n"
            "Download golfpose_17+2_33.6_30.5_59.5.bin from the GolfPose-3D table in "
            "https://github.com/MingHanLee/GolfPose and place it there."
        )
    if str(GOLFPOSE_DIR) not in sys.path:
        sys.path.insert(0, str(GOLFPOSE_DIR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from common.model_cross import MixSTE2

    model = MixSTE2(
        num_frame=RECEPTIVE_FIELD,
        num_joints=len(GOLFPOSE_JOINTS),
        in_chans=2,
        embed_dim_ratio=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_pos"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.eval()


def swap_sides(tensor: torch.Tensor) -> torch.Tensor:
    swapped = tensor.clone()
    swapped[:, :, JOINTS_LEFT + JOINTS_RIGHT] = tensor[:, :, JOINTS_RIGHT + JOINTS_LEFT]
    return swapped


def lift_window(model: torch.nn.Module, window: np.ndarray) -> np.ndarray:
    """One 243-frame window, averaged with its mirror as in GolfPose evaluation."""
    inputs = torch.from_numpy(window[None])
    mirrored = swap_sides(inputs)
    mirrored[..., 0] *= -1
    with torch.inference_mode():
        predicted = model(torch.cat([inputs, mirrored]))
    unmirrored = predicted[1:2].clone()
    unmirrored[..., 0] *= -1
    unmirrored = swap_sides(unmirrored)
    pose = ((predicted[0:1] + unmirrored) / 2.0)[0]
    return (pose - pose[:, :1]).numpy()


def lift_sequence(model: torch.nn.Module, normalized: np.ndarray) -> np.ndarray:
    count = len(normalized)
    if count <= RECEPTIVE_FIELD:
        starts = [0]
    else:
        starts = list(range(0, count - RECEPTIVE_FIELD + 1, RECEPTIVE_FIELD))
        if starts[-1] + RECEPTIVE_FIELD < count:
            starts.append(count - RECEPTIVE_FIELD)

    output = np.zeros((count, normalized.shape[1], 3), dtype=np.float32)
    done = np.zeros(count, dtype=bool)
    for start in starts:
        window = normalized[start : start + RECEPTIVE_FIELD]
        length = len(window)
        if length < RECEPTIVE_FIELD:
            padding = np.repeat(window[-1:], RECEPTIVE_FIELD - length, axis=0)
            window = np.concatenate([window, padding])
        pose = lift_window(model, np.ascontiguousarray(window, dtype=np.float32))
        for offset in range(length):
            if not done[start + offset]:
                output[start + offset] = pose[offset]
                done[start + offset] = True
        print(f"  3D window frames {start}-{start + length - 1}", flush=True)
    return output


def to_plot(points: np.ndarray) -> np.ndarray:
    """Camera axes (x right, y down, z forward) to plot axes (x right, depth, up)."""
    return np.stack([points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def rgb01(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0


class Panel3D:
    def __init__(self, width: int, height: int, poses: np.ndarray):
        dpi = 100
        self.figure = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_position((0.0, 0.0, 1.0, 0.96))
        self.size = (width, height)
        finite = poses[np.isfinite(poses).all(axis=(1, 2))]
        self.radius = float(np.abs(to_plot(finite)).max()) * 1.05 if len(finite) else 1.0

    def render(self, pose: np.ndarray, path: np.ndarray, path_times: np.ndarray, show_club: bool) -> np.ndarray:
        if not np.isfinite(pose).all():
            blank = np.full((self.size[1], self.size[0], 3), 255, dtype=np.uint8)
            cv2.putText(blank, "outside swing", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
            return blank
        axes = self.axes
        axes.cla()
        axes.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
        for set_limit in (axes.set_xlim3d, axes.set_ylim3d, axes.set_zlim3d):
            set_limit(-self.radius, self.radius)
        axes.set_box_aspect((1, 1, 1), zoom=PANEL_ZOOM)
        axes.set_xticklabels([])
        axes.set_yticklabels([])
        axes.set_zticklabels([])
        axes.set_title("GolfPose-3D 17+2", fontsize=11)

        plotted = to_plot(pose)
        for a, b in BONES:
            if (a, b) == ("shaft", "hosel"):
                if not show_club:
                    continue
                color = (0.1, 0.1, 0.1)
            else:
                color = rgb01(BODY_RGB[bone_side(a, b)])
            segment = plotted[[JOINT[a], JOINT[b]]]
            axes.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=2.5)

        if len(path) >= 2:
            plotted_path = to_plot(path)
            segments = np.stack([plotted_path[:-1], plotted_path[1:]], axis=1)
            colors = [rgb01(phase_color_t(t)[::-1]) for t in path_times[:-1]]
            axes.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=2.0))
        if len(path):
            head = to_plot(path[-1])
            color = rgb01(phase_color_t(path_times[-1])[::-1])
            axes.scatter([head[0]], [head[1]], [head[2]], color=[color], s=30)

        self.canvas.draw()
        image = np.asarray(self.canvas.buffer_rgba())[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size)
        return image


def draw_skeleton_2d(image: np.ndarray, points: np.ndarray, valid: np.ndarray) -> None:
    height, width = image.shape[:2]
    thickness = max(2, width // 400)
    for a, b in BONES:
        if (a, b) == ("shaft", "hosel"):
            continue
        ia, ib = JOINT[a], JOINT[b]
        if not (valid[ia] and valid[ib]):
            continue
        r, g, bl = BODY_RGB[bone_side(a, b)]
        cv2.line(
            image,
            as_point(points[ia], width, height),
            as_point(points[ib], width, height),
            (bl, g, r),
            thickness,
            cv2.LINE_AA,
        )
    for index in range(BODY_JOINTS):
        if valid[index]:
            cv2.circle(image, as_point(points[index], width, height), thickness + 1, (255, 255, 255), -1, cv2.LINE_AA)


def draw_ball(image: np.ndarray, ball_xy: list[float] | None) -> None:
    """Mark the ball YOLOE found. The point is the address detection."""
    if ball_xy is None:
        return
    center = (int(round(ball_xy[0])), int(round(ball_xy[1])))
    cv2.circle(image, center, 8, (0, 255, 255), 2, cv2.LINE_AA)
    outlined_text(image, "ball", (center[0] + 12, center[1] - 8), 0.6)


def club_frames(swing: Swing2D) -> np.ndarray:
    return np.flatnonzero(swing.valid[:, JOINT["hosel"]])


def render_video(
    video: Path,
    swing: Swing2D,
    poses: np.ndarray,
    saved: Path,
    still: Path,
    phases,
    ball_xy: list[float] | None = None,
) -> int:
    width, height = swing.size
    panel = Panel3D(width, height, poses)
    detected = club_frames(swing)
    first = int(detected[0]) if len(detected) else None
    last = int(detected[-1]) if len(detected) else None
    capture = cv2.VideoCapture(str(video))
    writer = None
    written = 0
    combined = None
    impact_image = None
    try:
        for frame_index in range(swing.frames):
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height))
            so_far = [sample for sample in swing.samples if sample.frame <= frame_index]
            left = render_swing_plane(frame, so_far, frame_index, phases)
            draw_skeleton_2d(left, swing.points[frame_index], swing.valid[frame_index])
            draw_ball(left, ball_xy)

            if first is not None and frame_index >= first and np.isfinite(poses[frame_index]).all():
                stop_frame = min(frame_index, last)
                segment = poses[first : stop_frame + 1, JOINT["hosel"]]
                keep = np.isfinite(segment).all(axis=1)
                path = segment[keep]
                span = max(last - first, 1)
                path_times = (np.arange(first, stop_frame + 1) - first)[keep] / span
            else:
                path = np.zeros((0, 3))
                path_times = np.zeros(0)
            pose = poses[frame_index]
            if not np.isfinite(pose).all():
                path = np.zeros((0, 3))
                path_times = np.zeros(0)
            right = panel.render(pose, path, path_times, show_club=first is not None and np.isfinite(pose).all())

            combined = np.hstack([left, right])
            if writer is None:
                writer = open_writer(saved, swing.fps, combined)
            writer.write(combined)
            if phases.impact is not None and frame_index == phases.impact:
                impact_image = combined.copy()
            written += 1
            if frame_index % 30 == 0:
                print(f"  render frame {frame_index}", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    chosen = impact_image if impact_image is not None else combined
    if chosen is not None:
        cv2.imwrite(str(still), chosen)
    return written


def length_stats(poses: np.ndarray, frames: np.ndarray, a: str, b: str) -> dict | None:
    if len(frames) == 0:
        return None
    start = poses[frames, JOINT[a]]
    end = poses[frames, JOINT[b]]
    good = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    if not good.any():
        return None
    lengths = np.linalg.norm(start[good] - end[good], axis=-1)
    return {"mean_m": round(float(lengths.mean()), 4), "std_m": round(float(lengths.std()), 4)}


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def lead_name(handedness: str) -> str:
    return "left" if handedness == "right" else "right"


def load_saved_face(stem: str) -> tuple[dict | None, dict | None]:
    path = ROOT / "runs" / "swing_plane" / f"{stem}_swing_plane.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("p4_face"), data.get("p7_face")


def track_ball(video: Path, ball_model, swing: Swing2D, address: int | None, impact: int | None, lead: str) -> dict:
    empty = {"p1_xy": None, "p1_frame": None, "p7_clubhead_distance_px": None}
    if ball_model is None or address is None:
        return empty
    wanted = set(range(max(0, address - 8), min(swing.frames, address + 6)))
    if impact is not None:
        wanted.add(impact)
    foot = JOINT["left_foot" if lead == "left" else "right_foot"]
    capture = cv2.VideoCapture(str(video))
    found: list[tuple[int, np.ndarray]] = []
    try:
        for frame_index in range(swing.frames):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index not in wanted:
                continue
            result = ball_model.predict(frame, conf=BALL_CONFIDENCE, imgsz=640, verbose=False)[0]
            if result.masks is None or len(result.masks.xy) == 0:
                continue
            box = swing.boxes[frame_index]
            box_area = None
            if np.isfinite(box).all():
                box_area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
            best = None
            best_area = float("inf")
            for polygon in result.masks.xy:
                contour = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
                if len(contour) < 3:
                    continue
                area = abs(float(cv2.contourArea(contour.astype(np.float32))))
                if box_area is not None and area > BALL_AREA_FRACTION * box_area:
                    continue
                center = contour.mean(axis=0)
                if np.isfinite(box).all() and float(center[1]) < float((box[1] + box[3]) / 2.0):
                    continue
                if area < best_area:
                    best_area = area
                    best = center
            if best is not None:
                found.append((frame_index, best))
    finally:
        capture.release()
    if not found:
        return empty
    foot_point = swing.filled[address, foot]
    if not np.isfinite(foot_point).all():
        foot_point = None
    address_hits = [(frame_index, center) for frame_index, center in found if abs(frame_index - address) <= 8]
    if not address_hits:
        return empty
    if foot_point is not None:
        chosen = min(address_hits, key=lambda item: float(np.linalg.norm(item[1] - foot_point)))
    else:
        chosen = min(address_hits, key=lambda item: abs(item[0] - address))
    distance = None
    if impact is not None and bool(swing.valid[impact, JOINT["hosel"]]):
        distance = float(np.linalg.norm(swing.points[impact, JOINT["hosel"]] - chosen[1]))
    return {
        "p1_xy": [round(float(chosen[1][0]), 2), round(float(chosen[1][1]), 2)],
        "p1_frame": int(chosen[0]),
        "p7_clubhead_distance_px": None if distance is None else round(distance, 2),
    }


def impact_from_ball(swing: Swing2D, ball_xy: list[float] | None, top: int | None, fallback: int | None) -> int | None:
    """Impact is the detected clubhead closest to the ball after the top.

    Frames with no club detection are ignored. The speed peak is used only
    when no ball was found, and only if the club was detected on that frame.
    """
    hosel = JOINT["hosel"]

    def detected(frame: int | None) -> bool:
        return frame is not None and 0 <= frame < swing.frames and bool(swing.valid[frame, hosel])

    if ball_xy is None or top is None:
        return fallback if detected(fallback) else None
    ball = np.asarray(ball_xy, dtype=np.float64)
    best = None
    best_distance = float("inf")
    for frame in range(top + 1, swing.frames):
        if not detected(frame):
            continue
        distance = float(np.linalg.norm(swing.points[frame, hosel] - ball))
        if distance < best_distance:
            best_distance = distance
            best = frame
    if best is not None:
        return best
    return fallback if detected(fallback) else None


def save_position_stills(
    video: Path,
    swing: Swing2D,
    positions: dict[str, int | None],
    folder: Path,
    phases,
    ball_xy: list[float] | None,
) -> list[str]:
    wanted: dict[int, list[str]] = {}
    for key, frame in positions.items():
        if frame is not None:
            wanted.setdefault(int(frame), []).append(key)
    if not wanted:
        return []
    capture = cv2.VideoCapture(str(video))
    written = []
    for key, frame in positions.items():
        if frame is None:
            stale = folder / f"{key.upper()}.jpg"
            if stale.exists():
                stale.unlink()
    try:
        for frame_index in range(swing.frames):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index not in wanted:
                continue
            so_far = [sample for sample in swing.samples if sample.frame <= frame_index]
            image = render_swing_plane(frame, so_far, frame_index, phases)
            draw_skeleton_2d(image, swing.points[frame_index], swing.valid[frame_index])
            for key in wanted[frame_index]:
                labeled = image.copy()
                outlined_text(labeled, f"{key.upper()} {POSITION_NAMES[key]}", (16, labeled.shape[0] - 24), 0.8)
                draw_ball(labeled, ball_xy)
                path = folder / f"{key.upper()}.jpg"
                cv2.imwrite(str(path), labeled)
                written.append(path.name)
    finally:
        capture.release()
    return written


def process(
    video: Path,
    pose_model: YOLO,
    club_model: YOLO,
    lifter: torch.nn.Module,
    wrists: WristFallback | None = None,
    handedness: str = "right",
    view: str = "down_the_line",
    ball_model=None,
) -> None:
    print(f"Source: {video}", flush=True)
    swing = detect_2d(video, pose_model, club_model, wrists)
    club = swing.points[:, JOINT["hosel"]].copy()
    club[~swing.valid[:, JOINT["hosel"]]] = np.nan
    phases = detect_phases(club, swing.wrist_y, swing.fps)
    swing.filled = smooth_keypoints(swing.filled, swing.fps)
    print(f"Phases: {phases.as_dict()}", flush=True)
    normalized, canvas = golf_canvas(swing)
    start = phases.address if phases.address is not None else 0
    stop = phases.follow_through if phases.follow_through is not None else swing.frames - 1
    stop = min(max(stop, start), swing.frames - 1)
    lifted = lift_sequence(lifter, normalized[start : stop + 1])
    poses = np.full((swing.frames, len(GOLFPOSE_JOINTS), 3), np.nan, dtype=np.float32)
    poses[start : stop + 1] = lifted
    lead = lead_name(handedness)
    ball = track_ball(video, ball_model, swing, phases.address, None, lead)
    phases.impact = impact_from_ball(swing, ball["p1_xy"], phases.top, phases.impact)
    if phases.impact is not None and ball["p1_xy"] is not None and bool(swing.valid[phases.impact, JOINT["hosel"]]):
        ball["p7_clubhead_distance_px"] = round(
            float(np.linalg.norm(swing.points[phases.impact, JOINT["hosel"]] - np.asarray(ball["p1_xy"]))),
            2,
        )
    checkpoints = find_checkpoint_frames(poses, phases.address, phases.top, phases.impact, lead)
    p4_face, p7_face = load_saved_face(video.stem)
    wrist_angles = {
        "p4": wrist_shaft_degrees(poses, phases.top, lead),
        "p7": wrist_shaft_degrees(poses, phases.impact, lead),
    }

    folder = OUTPUT_DIR / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / f"{video.stem}_3d.avi"
    still = folder / f"{video.stem}_3d.jpg"
    arrays = folder / f"{video.stem}_3d.npz"
    summary = folder / f"{video.stem}_3d.json"
    written = render_video(video, swing, poses, saved, still, phases, ball["p1_xy"])
    stills = save_position_stills(video, swing, checkpoints, folder, phases, ball["p1_xy"])

    np.savez_compressed(
        arrays,
        joints=np.array(GOLFPOSE_JOINTS),
        keypoints_2d=swing.points.astype(np.float32),
        keypoints_2d_valid=swing.valid,
        keypoints_2d_filled=swing.filled.astype(np.float32),
        keypoints_2d_normalized=normalized,
        keypoints_3d=poses,
        fps=np.float32(swing.fps),
        frame_size=np.array(swing.size),
    )

    every_frame = np.arange(swing.frames)
    detected = club_frames(swing)
    both_club = np.flatnonzero(swing.valid[:, JOINT["shaft"]] & swing.valid[:, JOINT["hosel"]])
    summary.write_text(
        json.dumps(
            {
                "source": relative(video),
                "video": relative(saved),
                "still": relative(still),
                "arrays": relative(arrays),
                "checkpoint": relative(CHECKPOINT),
                "pose_model": POSE_MODEL,
                "club_model": relative(MODEL_PATH),
                "frames": swing.frames,
                "frames_written": written,
                "fps": round(swing.fps, 3),
                "receptive_field": RECEPTIVE_FIELD,
                "normalization": canvas,
                "handedness": handedness,
                "view": view,
                "lead_side": lead,
                "phases": phases.as_dict(),
                "positions": checkpoints,
                "position_stills": stills,
                "ball": ball,
                "p4_face": p4_face,
                "p7_face": p7_face,
                "projected_wrist_deg": {
                    key: None if value is None else round(value, 1) for key, value in wrist_angles.items()
                },
                "lift_window": [int(start), int(stop)],
                "club_detected_frames": int(len(detected)),
                "mediapipe_wrists_replaced": 0 if wrists is None else wrists.replaced,
                "interpolated_frames": {
                    name: int((~swing.valid[:, index]).sum()) for index, name in enumerate(GOLFPOSE_JOINTS)
                },
                "bone_length_3d": {
                    f"{a}-{b}": length_stats(poses, both_club if a == "shaft" else every_frame, a, b)
                    for a, b in BONES
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Frames: {swing.frames} (written {written})")
    print(f"Club detected in {len(detected)} frames")
    print(f"Phases: {phases.as_dict()}")
    print(f"Positions: {checkpoints}")
    print(f"Ball: {ball}")
    print(f"Video: {saved}")
    print(f"Still: {still}")
    print(f"Position stills: {folder}")
    print(f"Arrays: {arrays}")
    print(f"Summary: {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift golf swing videos to 3D with GolfPose-3D.")
    parser.add_argument("video", nargs="?", type=Path, help="Swing video. Defaults to a random file in user_swings.")
    parser.add_argument("--all", action="store_true", help="Process every video in user_swings.")
    parser.add_argument(
        "--mediapipe-wrists",
        action="store_true",
        help="Fill YOLO wrists below 0.25 confidence from MediaPipe Pose.",
    )
    parser.add_argument("--handedness", choices=("right", "left"), default="right")
    parser.add_argument("--view", choices=("down_the_line", "face_on"), default="down_the_line")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise SystemExit(f"Trained golf-club weights not found: {MODEL_PATH}")
    if args.all:
        videos = collect_videos(VIDEO_DIR)
        if not videos:
            raise SystemExit(f"No videos found in {VIDEO_DIR}")
    else:
        videos = [resolve_video(args.video)]

    lifter = load_lifter(CHECKPOINT)
    pose_model = YOLO(POSE_MODEL)
    club_model = YOLO(str(MODEL_PATH))
    ball_model = None
    if BALL_MODEL.is_file():
        ball_model = YOLOE(str(BALL_MODEL))
        ball_model.set_classes(["golf ball"])
    wrists = WristFallback() if args.mediapipe_wrists else None
    try:
        for video in videos:
            process(
                video,
                pose_model,
                club_model,
                lifter,
                wrists,
                handedness=args.handedness,
                view=args.view,
                ball_model=ball_model,
            )
    finally:
        if wrists is not None:
            wrists.close()


if __name__ == "__main__":
    main()
