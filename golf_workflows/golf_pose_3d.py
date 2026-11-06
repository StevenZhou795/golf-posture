"""Lift golf swings with GolfPose MixSTE2 17+2 or 17+5.

The joint 22-point GolfPose ViTPose predicts the body and club together in native
GolfPose order; 17+2 keeps its body, shaft and hosel. Legacy COCO-body and
separate club backends remain selectable.

Usage:
    python -m golf_workflows.golf_pose_3d
    python -m golf_workflows.golf_pose_3d path/to/swing.mp4
    python -m golf_workflows.golf_pose_3d --all
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, replace
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
from golf_workflows.swing_phases import RELIABLE_CLUB_CONFIDENCE, SwingPhases, WristFallback, detect_phases, detect_refined_phases, find_checkpoint_frames, smooth_keypoints, wrist_shaft_degrees
from golf_workflows.trace_swing_plane import (
    CLUB_CONFIDENCE,
    KEYPOINT_CONFIDENCE,
    POSE_MODEL,
    ClubSample,
    as_point,
    best_club,
    best_club_box,
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
CHECKPOINT_22 = ROOT / "golfpose_checkpoints" / "golfpose_17+5_39.2_32.3_62.8.bin"
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
EXTRA_CLUB_JOINTS = ["heel", "toe_down", "toe_up"]
GOLFPOSE_JOINTS_22 = GOLFPOSE_JOINTS + EXTRA_CLUB_JOINTS
CLUB_JOINTS = {"shaft", "hosel", *EXTRA_CLUB_JOINTS}
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
BONES_22 = BONES + [("hosel", "heel"), ("heel", "toe_down"), ("toe_down", "toe_up")]


def joint_names(count: int) -> list[str]:
    if count == 19:
        return GOLFPOSE_JOINTS
    if count == 22:
        return GOLFPOSE_JOINTS_22
    raise ValueError(f"Supported GolfPose joint counts are 19 and 22, got {count}")


def pad_keypoints(points: np.ndarray, valid: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the lifter: add unavailable landmarks unmeasured, or drop heel/toe for 17+2."""
    joint_names(count)
    if points.shape[-2] == count:
        return points.copy(), valid.copy()
    if points.shape[-2] == 22 and count == 19:
        return points[..., :19, :].copy(), valid[..., :19].copy()
    if points.shape[-2] != 19 or count != 22:
        raise ValueError(f"Cannot convert {points.shape[-2]} joints to {count}")
    missing = np.full((*points.shape[:-2], 3, 2), np.nan, dtype=points.dtype)
    flags = np.zeros((*valid.shape[:-1], 3), dtype=bool)
    return np.concatenate((points, missing), axis=-2), np.concatenate((valid, flags), axis=-1)


def swing_for_lifter(swing: Swing2D, lifter: torch.nn.Module) -> Swing2D:
    count = lifter.Spatial_pos_embed.shape[1]
    if swing.points.shape[1] == count:
        return swing
    points, valid = pad_keypoints(swing.points, swing.valid, count)
    scores = {}
    if count < swing.points.shape[1]:
        scores = dict(
            keypoint_confidence=None if swing.keypoint_confidence is None else swing.keypoint_confidence[:, :count],
            club_confidence=None if swing.club_confidence is None else swing.club_confidence[:, :count - BODY_JOINTS])
    return replace(swing, points=points, valid=valid, filled=fill_gaps(points, valid), **scores)

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
# A ball within this fraction of the torso length of its address spot has not moved.
BALL_SPOT_RADIUS = 0.12
# Camera pan/zoom allowance between address and the downswing, in torso lengths.
BALL_DRIFT = 0.5
# A teed ball in a wide shot is only a few pixels wide; 640 px inference misses it.
BALL_PRESENCE_IMAGE_SIZE = 1280
# Clubhead approach is extrapolated only from within this multiple of the contact distance.
APPROACH_NEAR = 2.0
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
    club_confidence: np.ndarray | None = None
    keypoint_confidence: np.ndarray | None = None
    detector_details: list[dict] | None = None
    predicted_shaft_crops: list[dict] | None = None
    body_fusion: dict | None = None


def bone_side(a: str, b: str) -> str:
    if a in CLUB_JOINTS or b in CLUB_JOINTS:
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
    # Experimental 17+5 input: absent head markers collapse to the hosel.
    # The original points stay NaN and their valid flags stay False.
    if points.shape[1] == 22:
        for index in range(19, 22):
            if not valid[:, index].any():
                filled[:, index] = filled[:, JOINT["hosel"]]
    return filled


def detect_2d(video: Path, pose_model: YOLO, club_model: YOLO | None, wrists: WristFallback | None = None,
              club_pose_model=None, track_club_path: bool = False, club_crop_model=None) -> Swing2D:
    if getattr(pose_model, "joint_pose", False) is True:
        return pose_model.detect_sequence(video, track_club_path=track_club_path)
    pose_results = pose_model.predict(source=str(video), imgsz=640, stream=True, verbose=False)
    rows: list[np.ndarray] = []
    flags: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    samples: list[ClubSample] = []
    swing_wrist_y: list[float] = []
    club_confidences: list[np.ndarray] = []
    body_confidences: list[np.ndarray] = []
    detector_rows = []
    predicted_crops: list[dict] = []
    from golf_workflows.club_tracking import ShaftPredictor
    shaft_predictor = ShaftPredictor(RELIABLE_CLUB_CONFIDENCE)
    size = (0, 0)
    for frame_index, pose_result in enumerate(pose_results):
        frame = pose_result.orig_img
        size = (frame.shape[1], frame.shape[0])
        diagonal = max(float(np.hypot(frame.shape[0], frame.shape[1])), 1.0)
        joint_count = len(GOLFPOSE_JOINTS_22) if club_pose_model is not None else len(GOLFPOSE_JOINTS)
        points = np.full((joint_count, 2), np.nan)
        valid = np.zeros(joint_count, dtype=bool)
        landmark_confidence = np.zeros(5, dtype=np.float32)
        # Derived joints (root, spine, thorax, neck, head) have no single score.
        body_confidence = np.full(joint_count, np.nan, dtype=np.float32)
        body_confidence[[JOINT[name] for name in DIRECT_FROM_COCO]] = 0.0
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
            for name, coco_index in DIRECT_FROM_COCO.items():
                body_confidence[JOINT[name]] = conf[coco_index]
            if pose_result.boxes is not None and person < len(pose_result.boxes):
                box = pose_result.boxes.xyxy[person].cpu().numpy().astype(np.float64)

        previous = samples[-1] if track_club_path and samples else None
        if club_crop_model is not None:
            detection = None
            official_club_box, _crop_details = club_crop_model.club_box(frame,
                result=getattr(pose_result, 'yolox_detection', None), person_box=box, anchor=anchor)
            detector_rows.append(dict(frame_index=frame_index,
                body=getattr(pose_result, 'detector_details', None), club=_crop_details))
        else:
            club_result = club_model.predict(
                frame, imgsz=640, conf=CLUB_CONFIDENCE, retina_masks=True,
                max_det=3, verbose=False,
            )[0]
            detection = best_club(
                club_result, club_model.names, anchor,
                None if previous is None else previous.clubhead, diagonal,
                return_box=club_pose_model is not None,
            )
            official_club_box = None
        grip = head = None
        face = None
        if club_pose_model is not None:
            club_box = official_club_box
            golfer_box_fallback = False
            if club_crop_model is None and detection is not None:
                _seg_grip, _seg_head, _confidence, face, club_box = detection
            elif club_crop_model is None:
                club_box = best_club_box(club_result, club_model.names, anchor, diagonal)
                if club_box is None and np.isfinite(box).all():
                    club_box = box
                    golfer_box_fallback = True
            elif club_box is None and np.isfinite(box).all():
                club_box = box
                golfer_box_fallback = True
                _crop_details['fallback_reason'] = 'no associated club box; refine from golfer box'
            landmarks = None
            if club_box is not None:
                inference = (club_pose_model.predict_coarse_to_fine if golfer_box_fallback
                             else club_pose_model.predict)
                landmarks, landmark_confidence, _details = inference(frame, club_box)
            # Weak or missing club: a crop around the shaft predicted from the last
            # confident detections, anchored at the current hands.
            weak = landmarks is None or float(np.min(landmark_confidence[:2])) < RELIABLE_CLUB_CONFIDENCE
            prediction = shaft_predictor.predict_box(frame_index, anchor, size) if weak else None
            if prediction is not None:
                shaft_box, shaft_details = prediction
                retry, retry_confidence, _retry_details = club_pose_model.predict(frame, shaft_box)
                improved = landmarks is None or float(np.min(retry_confidence[:2])) > float(np.min(landmark_confidence[:2]))
                shaft_details.update(kept=bool(improved), min_confidence=round(float(np.min(retry_confidence[:2])), 3))
                predicted_crops.append(dict(frame_index=frame_index, **shaft_details))
                if improved:
                    landmarks, landmark_confidence = retry, retry_confidence
                    club_box = shaft_box
            if landmarks is not None:
                shaft_predictor.observe(frame_index, landmarks[0], landmarks[1], float(np.min(landmark_confidence[:2])))
                good = landmark_confidence >= club_pose_model.confidence_threshold
                points[17:22] = landmarks
                valid[17:22] = good
                grip = landmarks[0] if good[0] else None
                head = landmarks[1] if good[1] else None
                confidence = float(np.mean(landmark_confidence[:2]))
        elif detection is not None:
            grip, head, confidence, face = detection
            landmark_confidence[:2] = confidence
        if track_club_path and head is not None:
            sample = (stabilize(grip, head, confidence, frame_index, previous, diagonal, face)
                      if head is not None else None)
            if sample is not None:
                samples.append(sample)
                if club_pose_model is None:
                    points[JOINT["hosel"]] = sample.clubhead
                    valid[JOINT["hosel"]] = True
                    if sample.grip is not None:
                        points[JOINT["shaft"]] = sample.grip
                        valid[JOINT["shaft"]] = True

        rows.append(points)
        flags.append(valid)
        boxes.append(box)
        swing_wrist_y.append(wrist_height(pose_result, person))
        club_confidences.append(landmark_confidence)
        if joint_count > BODY_JOINTS:
            body_confidence[BODY_JOINTS:] = landmark_confidence[:joint_count - BODY_JOINTS]
        body_confidences.append(body_confidence)
        if frame_index % 10 == 0:
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
        club_confidence=np.stack(club_confidences),
        keypoint_confidence=np.stack(body_confidences),
        detector_details=detector_rows or None,
        predicted_shaft_crops=predicted_crops,
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

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_pos"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    count = state["Spatial_pos_embed"].shape[1]
    joint_names(count)
    model = MixSTE2(
        num_frame=RECEPTIVE_FIELD,
        num_joints=count,
        in_chans=2,
        embed_dim_ratio=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0,
    )
    model.load_state_dict(state, strict=True)
    model.checkpoint_path = checkpoint
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


def lift_sequence(model: torch.nn.Module, normalized: np.ndarray, blend_windows: bool = False) -> np.ndarray:
    expected = model.Spatial_pos_embed.shape[1]
    if normalized.shape[1:] != (expected, 2):
        raise ValueError(f"Checkpoint expects {expected} 2D joints; got {normalized.shape}")
    if not len(normalized) or not np.isfinite(normalized).all():
        raise ValueError("Lifter input must contain finite, filled 2D coordinates")
    count = len(normalized)
    if count <= RECEPTIVE_FIELD:
        starts = [0]
    else:
        stride = RECEPTIVE_FIELD // 2 if blend_windows else RECEPTIVE_FIELD
        starts = list(range(0, count - RECEPTIVE_FIELD + 1, stride))
        if starts[-1] + RECEPTIVE_FIELD < count:
            starts.append(count - RECEPTIVE_FIELD)

    output = np.zeros((count, normalized.shape[1], 3), dtype=np.float32)
    done = np.zeros(count, dtype=bool)
    totals = np.zeros(count, dtype=np.float32)
    for start in starts:
        window = normalized[start : start + RECEPTIVE_FIELD]
        length = len(window)
        if length < RECEPTIVE_FIELD:
            padding = np.repeat(window[-1:], RECEPTIVE_FIELD - length, axis=0)
            window = np.concatenate([window, padding])
        pose = lift_window(model, np.ascontiguousarray(window, dtype=np.float32))
        if blend_windows:
            weights = np.minimum(np.arange(length)+1, np.arange(length,0,-1)).astype(np.float32)
            output[start:start+length] += pose[:length]*weights[:,None,None]
            totals[start:start+length] += weights
            print(f"  3D blended window frames {start}-{start + length - 1}", flush=True)
            continue
        for offset in range(length):
            if not done[start + offset]:
                output[start + offset] = pose[offset]
                done[start + offset] = True
        print(f"  3D window frames {start}-{start + length - 1}", flush=True)
    if blend_windows:
        output /= totals[:,None,None]
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
        self.axes.set_position((0.0, 0.0, 1.0, 0.90 if poses.shape[1] == 22 else 0.96))
        self.size = (width, height)
        self.joints = joint_names(poses.shape[1])
        self.joint = {name: index for index, name in enumerate(self.joints)}
        self.bones = BONES_22 if len(self.joints) == 22 else BONES
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
        axes.set_title("GolfPose-3D " + ("17+5" if len(self.joints) == 22 else "17+2"), fontsize=11)

        plotted = to_plot(pose)
        for a, b in self.bones:
            if a in CLUB_JOINTS or b in CLUB_JOINTS:
                if not show_club:
                    continue
                color = (0.1, 0.1, 0.1)
            else:
                color = rgb01(BODY_RGB[bone_side(a, b)])
            segment = plotted[[self.joint[a], self.joint[b]]]
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
    names = joint_names(len(points))
    indexes = {name: index for index, name in enumerate(names)}
    bones = BONES_22 if len(points) == 22 else BONES
    for a, b in bones:
        if (a, b) == ("shaft", "hosel"):
            continue
        ia, ib = indexes[a], indexes[b]
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
    for index in range(BODY_JOINTS, len(points)):
        if valid[index]:
            cv2.circle(image, as_point(points[index], width, height), thickness + 2, (0, 255, 255), -1, cv2.LINE_AA)


def draw_ball(image: np.ndarray, ball_xy: list[float] | None) -> None:
    """Mark the ball YOLOE found. The point is the address detection."""
    if ball_xy is None:
        return
    center = (int(round(ball_xy[0])), int(round(ball_xy[1])))
    cv2.circle(image, center, 8, (0, 255, 255), 2, cv2.LINE_AA)
    outlined_text(image, "ball", (center[0] + 12, center[1] - 8), 0.6)


def club_frames(swing: Swing2D) -> np.ndarray:
    return np.flatnonzero(video_club_evidence(swing) != "none")


def video_club_evidence(swing: Swing2D) -> np.ndarray:
    """Apply the reference complete-club/shaft fallback to each video frame."""
    points, valid = swing.points[:, 17:], swing.valid[:, 17:]
    observed = valid & np.isfinite(points).all(axis=2)
    scores = (swing.keypoint_confidence[:, 17:] if swing.keypoint_confidence is not None
              else swing.club_confidence[:, :points.shape[1]] if swing.club_confidence is not None else None)
    if scores is not None:
        observed &= np.isfinite(scores)
    complete = observed.all(axis=1)
    length = np.linalg.norm(points[:, 1] - points[:, 0], axis=1)
    confidence = (swing.keypoint_confidence[:, 17] if swing.keypoint_confidence is not None
                  else swing.club_confidence[:, 0] if swing.club_confidence is not None
                  else observed[:, 0].astype(float))
    shaft = observed[:, 0] & np.isfinite(confidence) & (confidence >= .10)
    evidence = np.full(swing.frames, 'none', dtype='<U14')
    evidence[~complete & shaft] = 'shaft_fallback'
    evidence[complete & (length >= 12)] = 'complete'
    return evidence


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
    evidence = video_club_evidence(swing)
    detected = np.flatnonzero(swing.valid[:, JOINT["hosel"]]) if swing.samples else []
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
            right = panel.render(pose, path, path_times,
                                 show_club=evidence[frame_index] != 'none' and np.isfinite(pose).all())
            if evidence[frame_index] == 'shaft_fallback':
                outlined_text(right, 'club: inferred endpoints/head', (12, right.shape[0] - 15), .5)

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


def ball_presence(video: Path, ball_model, swing: Swing2D, frames: range, ball_xy: list[float],
                  radius: float, drift: float) -> tuple[dict[int, bool], list[float] | None]:
    """Whether the ball is still at rest on its spot, per frame, and that spot.

    The camera may pan or zoom between address and impact, so the spot is
    re-anchored on the first detection within `drift` of the address position
    and then followed while the ball stays within `radius` of it.
    """
    if ball_model is None or ball_xy is None:
        return {}, None
    spot = None
    wanted = set(frames)
    present: dict[int, bool] = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for frame_index in range(max(wanted) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index not in wanted:
                continue
            result = ball_model.predict(frame, conf=BALL_CONFIDENCE, imgsz=BALL_PRESENCE_IMAGE_SIZE, verbose=False)[0]
            centres = ([np.asarray(polygon, dtype=np.float64).reshape(-1, 2).mean(axis=0)
                        for polygon in result.masks.xy if len(polygon) >= 3]
                       if result.masks is not None else [])
            anchor = np.asarray(ball_xy, dtype=np.float64) if spot is None else spot
            limit = drift if spot is None else radius
            near = [centre for centre in centres if float(np.linalg.norm(centre - anchor)) <= limit]
            if near:
                spot = min(near, key=lambda centre: float(np.linalg.norm(centre - anchor)))
            present[frame_index] = bool(near)
    finally:
        capture.release()
    return present, None if spot is None else [round(float(spot[0]), 2), round(float(spot[1]), 2)]


def approach_time(distances: dict[int, tuple[float, str]], last: int, near: float) -> float | None:
    """Fractional frame at which the detected clubhead reaches the ball.

    Extrapolates the last two consecutive detected clubhead-to-ball distances
    up to frame `last`, provided the clubhead is closing in and already within
    `near` pixels. Impact itself is usually too blurred to detect.
    """
    for frame in range(last, min(distances), -1):
        (now, kind), (before, kind_before) = distances[frame], distances[frame - 1]
        if kind == kind_before == "detected":
            if before > now and now <= near:
                return frame + now / (before - now)
            return None
    return None


def contact_frame(swing: Swing2D, ball_xy: list[float] | None, window: tuple[int, int], max_distance: float,
                  tracked: np.ndarray | None = None, tracked_penalty: float = 0.0,
                  presence: dict[int, bool] | None = None) -> tuple[int | None, str | None, dict]:
    """Moment of impact from the ball, inside `window`.

    1. Ball departure: the ball was at its spot and, from some frame on, never
       is again. At contact the clubhead covers the ball, so the contact frame is
       that first missing frame or the one before; the clubhead nearer the ball
       decides between them.
    2. Otherwise the clubhead nearest the ball (within `max_distance`).
    The clubhead is the detected hosel, or the tracked hosel plus
    `tracked_penalty` pixels where the club was not detected (blur at impact).
    """
    details: dict = {"window": list(window)}
    if ball_xy is None:
        return None, None, details
    ball = np.asarray(ball_xy, dtype=np.float64)
    hosel = JOINT["hosel"]
    lower, upper = window

    def clubhead_distance(frame: int) -> tuple[float, str]:
        if swing.valid[frame, hosel] and np.isfinite(swing.points[frame, hosel]).all():
            return float(np.linalg.norm(swing.points[frame, hosel] - ball)), "detected"
        if tracked is not None and np.isfinite(tracked[frame]).all():
            return float(np.linalg.norm(tracked[frame] - ball)) + tracked_penalty, "tracked"
        return float("inf"), "none"

    distances = {frame: clubhead_distance(frame) for frame in range(lower, upper + 1)}
    details["clubhead_distance_px"] = {str(f): (None if not np.isfinite(d) else round(d, 1), kind)
                                       for f, (d, kind) in distances.items()}
    if presence:
        checked = sorted(presence)
        seen = False
        departure = None
        for frame in checked:
            if presence[frame]:
                seen = True
                departure = None
            elif seen and departure is None and frame >= lower:
                departure = frame
        if departure is not None and departure <= upper and all(not presence[f] for f in checked if f >= departure):
            options = [f for f in (departure - 1, departure) if lower <= f <= upper]
            best = min(options, key=lambda f: distances[f][0])
            details.update(ball_departure_frame=int(departure),
                           ball_presence={str(f): bool(v) for f, v in presence.items()})
            if distances[best][0] <= max_distance:
                return int(best), "ball_departure", details
            # Blurred or misdetected club at impact: extrapolate the approach seen
            # just before it, else the last frame with the ball at rest.
            details["clubhead_unconfirmed"] = True
            reach = approach_time(distances, options[0], APPROACH_NEAR * max_distance) if len(options) == 2 else None
            if reach is not None:
                details["clubhead_reaches_ball_at"] = round(reach, 2)
                return int(min(departure, max(options[0], round(reach)))), "ball_departure", details
            return int(options[0]), "ball_departure", details
        details["ball_presence"] = {str(f): bool(v) for f, v in presence.items()}
    frame, (distance, kind) = min(distances.items(), key=lambda item: item[1][0])
    if np.isfinite(distance) and distance <= max_distance:
        return int(frame), ("clubhead_at_ball" if kind == "detected" else "tracked_clubhead_at_ball"), details
    return None, None, details


def impact_from_ball(swing: Swing2D, ball_xy: list[float] | None, top: int | None, fallback: int | None,
                     search_window: tuple[int,int] | None = None, max_distance: float | None = None) -> int | None:
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
    lower,upper=search_window if search_window is not None else (top+1,swing.frames-1)
    for frame in range(max(top+1,lower),min(swing.frames-1,upper)+1):
        if not detected(frame):
            continue
        distance = float(np.linalg.norm(swing.points[frame, hosel] - ball))
        if distance < best_distance:
            best_distance = distance
            best = frame
    if max_distance is not None and (best is None or best_distance > max_distance):
        return None
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


class SkeletonQualityError(ValueError):
    """The golfer's 2D skeleton is missing for too long inside the swing."""


# Root, hips, shoulders, elbows, wrists: what the phase and position rules read.
SKELETON_CORE = (0, 1, 4, 11, 12, 13, 14, 15, 16)
SKELETON_CORE_REQUIRED = 6
# Self-occlusion at the top (arms behind the head) can hide the arms for a few
# frames; a drifted body crop loses the golfer for a second or more.
SKELETON_MAX_GAP_SECONDS = 0.3
SKELETON_MAX_LOST_FRACTION = 0.15


def skeleton_lost_frames(valid: np.ndarray) -> np.ndarray:
    """Frames where fewer than 6 of the 9 core joints, or neither wrist, were detected."""
    core = valid[:, list(SKELETON_CORE)]
    return (core.sum(axis=1) < SKELETON_CORE_REQUIRED) | ~(valid[:, 13] | valid[:, 16])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def check_skeleton(swing: Swing2D, start: int, stop: int, folder: Path, label: str) -> dict:
    """Abort with SkeletonQualityError when the skeleton is lost inside [start, stop].

    Lost frames are interpolated otherwise, and a long gap makes every later
    position land on invented poses. The limit is a gap longer than
    max(3 frames, 0.3 s) or more than 15% of the window. Diagnostics are written
    to `skeleton_diagnostics.json` either way.
    """
    lost = skeleton_lost_frames(swing.valid)
    start, stop = max(0, int(start)), min(swing.frames - 1, int(stop))
    window = lost[start:stop + 1]
    gaps = [(a + start, b + start) for a, b in _runs(window)]
    longest = max((b - a + 1 for a, b in gaps), default=0)
    max_gap = max(3, int(round(SKELETON_MAX_GAP_SECONDS * swing.fps)))
    fraction = float(window.mean()) if len(window) else 0.0
    failed = longest > max_gap or fraction > SKELETON_MAX_LOST_FRACTION
    sources = {}
    for row in swing.detector_details or []:
        if row is None:
            continue
        frame = row.get("frame_index")
        body = row.get("body") if isinstance(row.get("body"), dict) else row
        if frame is not None and start <= frame <= stop and lost[frame]:
            source = body.get("body_crop_source") or body.get("crop_source")
            sources[str(source)] = sources.get(str(source), 0) + 1
    report = {
        "checked_window": [start, stop], "window_label": label,
        "lost_frames": int(window.sum()), "lost_fraction": round(fraction, 3),
        "lost_ranges": [[int(a), int(b)] for a, b in gaps],
        "longest_gap_frames": int(longest), "max_gap_frames": max_gap,
        "max_lost_fraction": SKELETON_MAX_LOST_FRACTION, "crop_sources_on_lost_frames": sources,
        "core_joints_valid_per_frame": swing.valid[:, list(SKELETON_CORE)].sum(axis=1).tolist(),
        "passed": not failed,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "skeleton_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        ranges = ", ".join(f"{a}-{b} ({(b - a + 1) / swing.fps:.2f} s)" for a, b in gaps[:6])
        raise SkeletonQualityError(
            f"Golfer skeleton lost inside the {label} (frames {start}-{stop}): {int(window.sum())} frames "
            f"({fraction:.0%}), longest gap {longest} frames (limit {max_gap}). Lost ranges: {ranges}. "
            f"Positions in or after these frames would be chosen from interpolated poses, so the run was aborted. "
            f"Check the body crops on those frames ({folder / 'skeleton_diagnostics.json'}), supply --golfer-box "
            f"around the golfer, or pass --allow-skeleton-gaps to continue anyway.")
    return report


def process(
    video: Path,
    pose_model: YOLO,
    club_model: YOLO,
    lifter: torch.nn.Module,
    wrists: WristFallback | None = None,
    handedness: str = "right",
    view: str = "down_the_line",
    ball_model=None,
    reviewed_positions: dict[str, int] | None = None,
    club_refinement: bool = False,
    club_marker_length_m: float | None = None,
    output_folder: Path | None = None,
    club_pose_model=None,
    track_club_path: bool = False,
    club_crop_model=None,
    club_mask_model=None,
    allow_skeleton_gaps: bool = False,
) -> None:
    print(f"Source: {video}", flush=True)
    detected_2d = detect_2d(video, pose_model, club_model, wrists, club_pose_model, track_club_path, club_crop_model)
    swing = swing_for_lifter(detected_2d, lifter)
    folder = output_folder or OUTPUT_DIR / video.stem
    names = joint_names(swing.points.shape[1])
    dropped_2d = joint_names(detected_2d.points.shape[1])[len(names):] if detected_2d.points.shape[1] > len(names) else []
    if reviewed_positions is not None:
        if set(reviewed_positions) != set(POSITION_NAMES):
            raise ValueError("Reviewed positions must contain P1 through P10")
        if any(not 0 <= frame < swing.frames for frame in reviewed_positions.values()):
            raise ValueError("Reviewed frame selection exceeds video bounds")
    from golf_workflows.swing_phases import observed_club_track
    club, club_track_source = observed_club_track(swing.points, swing.valid)
    phase_inputs=swing.filled.copy()
    swing.filled = smooth_keypoints(swing.filled, swing.fps)
    club_landmark_confidence = (swing.club_confidence if swing.club_confidence is not None
                                else np.zeros((swing.frames, 5), dtype=np.float32))
    club_confidence = np.mean(club_landmark_confidence[:, :2], axis=1)
    if club_pose_model is None:
        for sample in swing.samples:
            club_confidence[sample.frame] = sample.confidence
    from golf_workflows.position_selection import infer_handedness, infer_view
    view_detection = {"source": "argument"}
    handedness_detection = {"source": "argument"}
    if view == "auto":
        view, view_detection = infer_view(swing.points, swing.valid)
    if handedness == "auto":
        handedness, handedness_detection = infer_handedness(swing.points, swing.valid, view)
    print(f"View: {view} ({view_detection['source']}), handedness: {handedness} ({handedness_detection['source']})",
          flush=True)
    club_tracking = {"applied": False, "reason": "club refinement disabled"}
    shaft_measurements = None
    track = None
    if club_refinement:
        from golf_workflows.club_refinement import prepare_club_inputs
        swing.filled, _ = prepare_club_inputs(swing.points, swing.filled, swing.valid, club_confidence)
        from golf_workflows.club_measurements import shaft_angle_measurements
        from golf_workflows.club_tracking import apply_track, track_club
        floor = (club_landmark_confidence[:, :2].min(axis=1) if swing.club_confidence is not None
                 else club_confidence)
        hands = (swing.filled[:, JOINT["left_wrist"]] + swing.filled[:, JOINT["right_wrist"]]) / 2.0
        detected_pair = swing.valid[:, JOINT["shaft"]] & swing.valid[:, JOINT["hosel"]]
        reliable_pair = detected_pair & (floor >= RELIABLE_CLUB_CONFIDENCE)
        lengths = np.linalg.norm(swing.points[:, JOINT["hosel"]] - swing.points[:, JOINT["shaft"]], axis=1)
        length_px = float(np.median(lengths[reliable_pair])) if reliable_pair.any() else float("nan")
        measurements, shaft_measurements = shaft_angle_measurements(
            video, set(int(f) for f in np.flatnonzero(~reliable_pair)), hands, length_px, club_mask_model)
        track = track_club(swing.points, swing.valid, floor, hands, swing.fps, measurements, RELIABLE_CLUB_CONFIDENCE)
        if track is None:
            club_tracking = {"applied": False, "reason": "fewer than two club detections"}
        else:
            swing.points, swing.valid = apply_track(swing.points, swing.valid, track)
            swing.filled[:, JOINT["shaft"]:JOINT["hosel"] + 1] = track.points
            phase_inputs[:, JOINT["shaft"]:JOINT["hosel"] + 1] = track.points
            club_confidence[track.rejected] = 0.0
            club_landmark_confidence = club_landmark_confidence.copy()
            club_landmark_confidence[track.rejected, :2] = 0.0
            club_tracking = {"applied": True, **track.summary}
    phase_diagnostics={"method":"legacy_clubhead_or_wrist", "club_track_source": club_track_source}
    if reviewed_positions is not None:
        phases=SwingPhases(reviewed_positions["p1"],reviewed_positions["p4"],reviewed_positions["p7"],
                          reviewed_positions["p10"],"visual_review")
        phase_diagnostics={"method":"reviewed_indices"}
    elif club_refinement:
        phases,phase_diagnostics=detect_refined_phases(swing.points,swing.valid,phase_inputs,swing.fps,club_confidence)
    else:
        phases=detect_phases(club,swing.wrist_y,swing.fps)
    # Ball correction must precede window selection and lifting. Refined phases
    # restrict its search to the corroborated wrist return, not the whole clip.
    lead = lead_name(handedness)
    ball = track_ball(video,ball_model,swing,phases.address,phases.impact,lead)
    contact = None
    if reviewed_positions is None:
        if club_refinement and phase_diagnostics.get('wrist_impact') is not None:
            from golf_workflows.swing_phases import impact_window
            window = impact_window(phase_diagnostics['wrist_impact'], phases.top, swing.fps, swing.frames)
            scale_px = phase_diagnostics['body_scale_px']
            presence, spot = {}, ball['p1_xy']
            if ball['p1_xy'] is not None:
                lead_in = max(3, round(.3 * swing.fps))
                presence, spot = ball_presence(video, ball_model, swing,
                                               range(max(phases.top, window[0] - lead_in), window[1] + 1),
                                               ball['p1_xy'], BALL_SPOT_RADIUS * scale_px, BALL_DRIFT * scale_px)
                spot = spot or ball['p1_xy']
            frame, source, contact = contact_frame(
                swing, spot, window, .6 * scale_px,
                tracked=None if track is None else track.points[:, 1], tracked_penalty=.1 * scale_px,
                presence=presence)
            contact.update(source=source, ball_spot=spot)
            if frame is not None:
                phases,phase_diagnostics=detect_refined_phases(swing.points,swing.valid,phase_inputs,swing.fps,
                                                              club_confidence,frame,source)
        elif not club_refinement:
            corrected=impact_from_ball(swing,ball['p1_xy'],phases.top,phases.impact)
            phases.impact=corrected
            if corrected is not None:
                # Retain uncertain tail instead of clipping before corrected contact.
                phases.follow_through=max(phases.follow_through or 0,min(swing.frames-1,corrected+round(.5*swing.fps)))
    print(f"Phases: {phases.as_dict()}", flush=True)
    if phase_diagnostics.get('impact_source'):
        print(f"P7 source: {phase_diagnostics['impact_source']} (window {phase_diagnostics.get('impact_window')})", flush=True)
    skeleton_check = {"skipped": "no swing window"}
    if phases.address is not None:
        window_stop = phases.follow_through if phases.follow_through is not None else swing.frames - 1
        try:
            skeleton_check = check_skeleton(swing, phases.address, window_stop, folder, "swing window (P1 to finish)")
        except SkeletonQualityError as error:
            if not allow_skeleton_gaps:
                print(f"ERROR: {error}", flush=True)
                raise
            print(f"WARNING (--allow-skeleton-gaps): {error}", flush=True)
            skeleton_check = json.loads((folder / "skeleton_diagnostics.json").read_text(encoding="utf-8"))
            skeleton_check["overridden"] = True
    normalized, canvas = golf_canvas(swing)
    start = phases.address if phases.address is not None else 0
    stop = phases.follow_through if phases.follow_through is not None else swing.frames - 1
    stop = min(max(stop, start), swing.frames - 1)
    lifted = lift_sequence(lifter, normalized[start : stop + 1], blend_windows=club_refinement)
    poses = np.full((swing.frames, len(names), 3), np.nan, dtype=np.float32)
    poses[start : stop + 1] = lifted
    unrefined_poses = poses.copy()
    refinement = {"applied": False, "reason": "disabled"}
    if club_refinement:
        from golf_workflows.club_refinement import refine_club_3d
        poses, refinement = refine_club_3d(poses, normalized, swing.valid, club_confidence, club_marker_length_m)
    if phases.impact is not None and ball["p1_xy"] is not None and bool(swing.valid[phases.impact, JOINT["hosel"]]):
        ball["p7_clubhead_distance_px"] = round(
            float(np.linalg.norm(swing.points[phases.impact, JOINT["hosel"]] - np.asarray(ball["p1_xy"]))),
            2,
        )
    club_observed = None
    position_sources = {}
    if club_refinement:
        landmark_floor = (club_landmark_confidence[:, :2].min(axis=1) if swing.club_confidence is not None
                          else club_confidence)
        club_observed = (swing.valid[:, JOINT["shaft"]] & swing.valid[:, JOINT["hosel"]]
                         & (landmark_floor >= RELIABLE_CLUB_CONFIDENCE))
    selection = {"method": "rule_based"}
    if reviewed_positions is not None:
        checkpoints = reviewed_positions.copy()
    else:
        rule_based = find_checkpoint_frames(poses, phases.address, phases.top, phases.impact, lead,
                                            reserve_positions=club_refinement, club_observed=club_observed,
                                            sources=position_sources)
        checkpoints = rule_based
        if club_refinement:
            from golf_workflows.position_selection import ground_roll, level_poses, select_positions, subframe_ms
            roll, leveling = ground_roll(swing.points, swing.valid, view)
            image_points = None
            if view == "face_on":
                observed_2d = np.where(swing.valid[..., None], swing.points, np.nan).astype(np.float64)
                padded = np.concatenate([observed_2d, np.zeros(observed_2d.shape[:2] + (1,))], axis=2)
                image_points = level_poses(padded, roll)
            selected, selection = select_positions(
                level_poses(poses, roll), phases.address, phases.top, phases.impact, lead,
                club_observed=club_observed, confidence=swing.keypoint_confidence, image_points=image_points)
            if selection["method"] == "joint_dp":
                checkpoints = selected
                position_sources = selection.pop("sources")
                selection["subframe_ms"] = subframe_ms(selection["subframe"], swing.fps)
                selection["rule_based_positions"] = rule_based
            selection["ground_leveling"] = leveling
    p4_face, p7_face = load_saved_face(video.stem)
    wrist_angles = {
        "p4": wrist_shaft_degrees(poses, phases.top, lead),
        "p7": wrist_shaft_degrees(poses, phases.impact, lead),
    }

    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / f"{video.stem}_3d.avi"
    still = folder / f"{video.stem}_3d.jpg"
    arrays = folder / f"{video.stem}_3d.npz"
    summary = folder / f"{video.stem}_3d.json"
    written = render_video(video, swing, poses, saved, still, phases, ball["p1_xy"])
    stills = save_position_stills(video, swing, checkpoints, folder, phases, ball["p1_xy"])

    np.savez_compressed(
        arrays,
        joints=np.array(names),
        keypoints_2d=swing.points.astype(np.float32),
        keypoints_2d_valid=swing.valid,
        keypoints_2d_filled=swing.filled.astype(np.float32),
        keypoints_2d_normalized=normalized,
        keypoints_3d=poses,
        keypoints_3d_before_club_refinement=unrefined_poses,
        club_confidence=club_confidence,
        club_landmark_confidence=club_landmark_confidence,
        club_evidence=video_club_evidence(swing),
        **({"keypoints_2d_confidence": swing.keypoint_confidence}
           if swing.keypoint_confidence is not None else {}),
        **({"club_track_points": track.points.astype(np.float32),
            "club_track_angle": track.theta.astype(np.float32),
            "club_track_sigma_deg": track.sigma_deg.astype(np.float32),
            "club_track_accepted": track.accepted, "club_track_rejected": track.rejected,
            "club_track_swapped": track.swapped, "club_track_measurement_used": track.measurement_used}
           if track is not None else {}),
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
                "checkpoint": relative(lifter.checkpoint_path),
                "joint_count": len(names),
                "unmeasured_extra_club_inputs": (EXTRA_CLUB_JOINTS if len(names) == 22 and club_pose_model is None else []),
                "extra_club_input_fill": ("hosel" if len(names) == 22 and club_pose_model is None else None),
                "dropped_2d_landmarks": dropped_2d,
                "pose_model": getattr(pose_model,"metadata",{}).get("pose_checkpoint",POSE_MODEL),
                "pose_backend": getattr(pose_model,"metadata",{}),
                "club_model": relative(MODEL_PATH) if club_model is not None else None,
                "club_pose_backend": (getattr(club_pose_model, "metadata", None)
                                      if club_pose_model is not None else {"club_pose_backend": "segmentation"}),
                "club_crop_backend": ({"club_crop_backend": pose_model.metadata.get("club_crop_backend"),
                                       "detector": pose_model.metadata.get("person_detector")}
                                      if getattr(pose_model, "joint_pose", False) is True else
                                      getattr(club_crop_model, "metadata", None)
                                      if club_crop_model is not None else {"club_crop_backend": "segmentation"}),
                "club_path_tracking": track_club_path,
                "detector_crops": swing.detector_details,
                "club_path_rendering": track_club_path,
                "club_refinement": refinement,
                "window_blending": club_refinement,
                "frames": swing.frames,
                "frames_written": written,
                "fps": round(swing.fps, 3),
                "receptive_field": RECEPTIVE_FIELD,
                "normalization": canvas,
                "handedness": handedness,
                "view": view,
                "lead_side": lead,
                "phases": phases.as_dict(),
                "phase_detection": phase_diagnostics,
                "positions": checkpoints,
                "skeleton_check": {key: value for key, value in skeleton_check.items()
                                   if key != "core_joints_valid_per_frame"},
                "view_detection": view_detection,
                "handedness_detection": handedness_detection,
                "club_tracking": club_tracking,
                "shaft_angle_measurements": shaft_measurements,
                "predicted_shaft_crops": {
                    "tried": len(swing.predicted_shaft_crops or []),
                    "kept": sum(1 for row in swing.predicted_shaft_crops or [] if row.get("kept")),
                    "frames": swing.predicted_shaft_crops or [],
                },
                "body_fusion": swing.body_fusion,
                "checkpoint_selection": {"method": "reviewed_indices_reused" if reviewed_positions is not None else "automatic",
                                         "selector": selection.get("method"),
                                         "position_confidence": selection.get("confidence"),
                                         "subframe_frames": selection.get("subframe"),
                                         "subframe_ms": selection.get("subframe_ms"),
                                         "rule_based_positions": selection.get("rule_based_positions"),
                                         "ground_leveling": selection.get("ground_leveling"),
                                         "trail_wrist_frames": selection.get("trail_wrist_frames"),
                                         "position_sources": position_sources,
                                         "reliable_club_confidence": RELIABLE_CLUB_CONFIDENCE if club_observed is not None else None,
                                         "reliable_club_frames": None if club_observed is None else int(club_observed.sum()),
                                         "frame_index_base": 0, "fresh_2d_inference": True},
                "position_stills": stills,
                "ball": ball,
                "impact_contact": contact,
                "p4_face": p4_face,
                "p7_face": p7_face,
                "projected_wrist_deg": {
                    key: None if value is None else round(value, 1) for key, value in wrist_angles.items()
                },
                "lift_window": [int(start), int(stop)],
                "club_detected_frames": int(len(detected)),
                "club_evidence_policy": "complete club first; otherwise confident shaft with inferred MixSTE geometry",
                "club_evidence_counts": {kind: int((video_club_evidence(swing) == kind).sum())
                                         for kind in ('complete', 'shaft_fallback', 'none')},
                "mediapipe_wrists_replaced": 0 if wrists is None else wrists.replaced,
                "interpolated_frames": {
                    name: int((~swing.valid[:, index]).sum()) for index, name in enumerate(names)
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


def load_reviewed_positions(path: Path, video: Path) -> dict[str, int]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("frame_index_base", 0) != 0:
        raise ValueError("Reviewed indices must be zero-based")
    if selection.get("source") and Path(selection["source"]).name != video.name:
        raise ValueError("Frame-selection source does not match the input video")
    rows = selection["frames"]
    if len(rows) != 10:
        raise ValueError("Expected exactly ten reviewed frame records")
    positions = {row["position"].lower(): row["frame_index"] for row in rows}
    if set(positions) != set(POSITION_NAMES):
        raise ValueError("Frame records must cover P1 through P10 once each")
    indices = [positions[f"p{i}"] for i in range(1, 11)]
    if any(type(i) is not int or i < 0 for i in indices) or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Reviewed indices must be nonnegative, strictly increasing integers")
    return positions


def main(argv: list[str] | None = None) -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Lift golf swing videos to 3D with GolfPose-3D.")
    parser.add_argument("video", nargs="?", type=Path, help="Swing video. Defaults to a random file in user_swings.")
    parser.add_argument("--all", action="store_true", help="Process every video in user_swings.")
    parser.add_argument("--lifter", choices=("17+2", "17+5"), default="17+2")
    parser.add_argument("--pose-backend", choices=("yolo26n-pose", "vitpose-huge", "golfpose-golfer-vitpose-huge"), default="yolo26n-pose")
    parser.add_argument("--pose-cpu-threads", type=int, default=4)
    parser.add_argument("--golfer-box", type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help="Golfer ROI in original-frame pixels, used where the GolfPose detector misses "
                             "(joint ViTPose, or YOLO26n with the YOLOX body crop). Overrides the automatic ROI.")
    parser.add_argument("--no-auto-golfer-box", action="store_true",
                        help="Do not derive a default golfer ROI from a YOLO26n pass over sampled frames.")
    parser.add_argument("--club-pose-backend", choices=("segmentation", "vitpose-huge"), default="vitpose-huge")
    parser.add_argument("--club-crop-backend", choices=("segmentation", "official-yolox-2cls"), default="official-yolox-2cls")
    parser.add_argument("--track-club-path", action="store_true",
                        help="Opt in to temporal club association and the rendered swing trail.")
    parser.add_argument("--positions-json", type=Path, help="Reuse reviewed P1-P10 indices while inferring new poses.")
    parser.add_argument("--club-refinement", action="store_true", help="Use angular gap filling, blended windows and constrained club geometry.")
    parser.add_argument("--club-marker-length-m", type=float, help="Measured shaft-marker to hosel-marker distance, used with --club-refinement.")
    parser.add_argument("--output-dir", type=Path, help="Separate output folder; 17+5 uses a separate default.")
    parser.add_argument("--run-dir", type=Path, help="Exact single-video output folder (used by workflow.py).")
    parser.add_argument(
        "--mediapipe-wrists",
        action="store_true",
        help="Fill YOLO wrists below 0.25 confidence from MediaPipe Pose.",
    )
    parser.add_argument("--handedness", choices=("auto", "right", "left"), default="auto",
                        help="auto: from the 2D hands (falls back to right when ambiguous).")
    parser.add_argument("--view", choices=("auto", "down_the_line", "face_on"), default="auto",
                        help="auto: from shoulder width and the feet line at address (falls back to down_the_line).")
    parser.add_argument("--no-body-fusion", action="store_true",
                        help="Joint ViTPose only: do not fill low-confidence body joints from YOLO26n.")
    parser.add_argument("--no-club-mask", action="store_true",
                        help="Do not use the golf-club segmentation mask as a shaft-angle measurement.")
    parser.add_argument("--allow-skeleton-gaps", action="store_true",
                        help="Warn instead of aborting when the golfer skeleton is lost inside the swing window.")
    args = parser.parse_args(argv)
    joint_pose = args.pose_backend == "golfpose-golfer-vitpose-huge"
    roi_capable = joint_pose or (args.pose_backend == "yolo26n-pose" and args.club_crop_backend == "official-yolox-2cls")
    if args.golfer_box is not None and not roi_capable:
        parser.error('--golfer-box applies to --pose-backend golfpose-golfer-vitpose-huge, '
                     'or yolo26n-pose with --club-crop-backend official-yolox-2cls')
    if joint_pose and args.mediapipe_wrists:
        parser.error("--mediapipe-wrists applies only to the legacy COCO body backends")
    if args.run_dir and args.all:
        parser.error("--run-dir accepts a single video only")
    if not joint_pose and args.club_crop_backend == "official-yolox-2cls" and args.club_pose_backend != "vitpose-huge":
        parser.error("--club-crop-backend official-yolox-2cls requires --club-pose-backend vitpose-huge")
    if args.club_marker_length_m is not None and (not args.club_refinement or not np.isfinite(args.club_marker_length_m) or args.club_marker_length_m <= 0):
        parser.error("--club-marker-length-m requires --club-refinement and a finite positive distance")
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.resolve()
    elif joint_pose:
        OUTPUT_DIR = ROOT / "runs" / ("golf_pose_3d_golfer_vitpose_" + args.lifter.replace("+", "plus"))
    elif args.pose_backend == "vitpose-huge":
        OUTPUT_DIR = ROOT / "runs" / ("golf_pose_3d_vitpose_huge_" + args.lifter.replace("+", "plus"))
    elif args.lifter == "17+5":
        OUTPUT_DIR = ROOT / "runs" / "golf_pose_3d_17plus5"
    if args.club_refinement and args.output_dir is None:
        OUTPUT_DIR = OUTPUT_DIR.with_name(OUTPUT_DIR.name + "_club_refined")

    needs_segmentation = not joint_pose and (args.club_crop_backend == "segmentation" or args.club_pose_backend == "segmentation")
    if needs_segmentation and not MODEL_PATH.is_file():
        raise SystemExit(f"Trained golf-club weights not found: {MODEL_PATH}")
    if args.all:
        videos = collect_videos(VIDEO_DIR)
        if not videos:
            raise SystemExit(f"No videos found in {VIDEO_DIR}")
    else:
        videos = [resolve_video(args.video)]

    lifter = load_lifter(CHECKPOINT_22 if args.lifter == "17+5" else CHECKPOINT)
    from golf_workflows.vitpose_tracking import load_pose_backend
    pose_model = load_pose_backend(args.pose_backend, cpu_threads=args.pose_cpu_threads)
    if args.golfer_box is not None:
        from golf_workflows.golfer_vitpose import validate_golfer_box
        try:
            validate_golfer_box(args.golfer_box)
        except ValueError as exc:
            parser.error(str(exc))
    club_model = YOLO(str(MODEL_PATH)) if needs_segmentation else None
    from golf_workflows.club_vitpose import load_club_pose_backend
    try:
        club_pose_model = pose_model if joint_pose else load_club_pose_backend(args.club_pose_backend)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    from golf_workflows.official_yolox import load_club_crop_backend
    try:
        club_crop_model = None if joint_pose else load_club_crop_backend(args.club_crop_backend)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.pose_backend == 'yolo26n-pose' and club_crop_model is not None:
        from golf_workflows.detector_pose import YOLOXBodyPose
        pose_model = YOLOXBodyPose(pose_model, club_crop_model)
    ball_model = None
    if BALL_MODEL.is_file():
        ball_model = YOLOE(str(BALL_MODEL))
        ball_model.set_classes(["golf ball"])
    wrists = WristFallback() if args.mediapipe_wrists else None
    roi_model = None
    from golf_workflows.golfer_roi import POSE_WEIGHTS
    if roi_capable and args.golfer_box is None and not args.no_auto_golfer_box:
        roi_model = pose_model.pose_model if hasattr(pose_model, "pose_model") else YOLO(str(POSE_WEIGHTS))
    if joint_pose and not args.no_body_fusion and POSE_WEIGHTS.is_file():
        pose_model.body_fallback_model = roi_model or YOLO(str(POSE_WEIGHTS))
        pose_model.metadata["body_fusion"] = "yolo26n-pose where ViTPose < 0.3 and YOLO26n >= 0.5"
    club_mask_model = None
    if args.club_refinement and not args.no_club_mask and MODEL_PATH.is_file():
        club_mask_model = club_model or YOLO(str(MODEL_PATH))
    try:
        for video in videos:
            if joint_pose:
                pose_model.diagnostics_dir = args.run_dir.resolve() if args.run_dir else OUTPUT_DIR / video.stem
            if roi_capable:
                if args.golfer_box is not None:
                    pose_model.set_initial_golfer_box(args.golfer_box, "manual")
                elif roi_model is not None:
                    from golf_workflows.golfer_roi import auto_golfer_roi
                    roi, roi_details = auto_golfer_roi(video, roi_model)
                    print(f"Golfer ROI (auto YOLO26n): {roi_details['roi_xyxy']}", flush=True)
                    pose_model.set_initial_golfer_box(roi, "auto-yolo26n", roi_details)
                else:
                    pose_model.set_initial_golfer_box(None, details={"source": "disabled"})
            process(
                video,
                pose_model,
                club_model,
                lifter,
                wrists,
                handedness=args.handedness,
                view=args.view,
                ball_model=ball_model,
                reviewed_positions=load_reviewed_positions(args.positions_json, video) if args.positions_json else None,
                club_refinement=args.club_refinement,
                club_marker_length_m=args.club_marker_length_m,
                output_folder=args.run_dir.resolve() if args.run_dir else None,
                club_pose_model=club_pose_model,
                track_club_path=args.track_club_path,
                club_crop_model=club_crop_model,
                club_mask_model=club_mask_model,
                allow_skeleton_gaps=args.allow_skeleton_gaps,
            )
    finally:
        if wrists is not None:
            wrists.close()


if __name__ == "__main__":
    main()
