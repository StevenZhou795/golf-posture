"""Lift the ten standard position sheets into one 3D reference pose each.

Each sheet shows the same position from face-on, down-the-line, and rear.
The three lifts are aligned to the down-the-line view and averaged.

Usage:
    python standard_frames.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from golf_pose_3d import (
    BONES,
    BODY_JOINTS,
    GOLFPOSE_JOINTS,
    JOINT,
    JOINTS_LEFT,
    JOINTS_RIGHT,
    CHECKPOINT,
    Panel3D,
    Swing2D,
    coco_to_golfpose,
    draw_skeleton_2d,
    fill_gaps,
    golf_canvas,
    lift_sequence,
    load_lifter,
)
from predict_seg import MODEL_PATH
from trace_swing_plane import (
    CLUB_CONFIDENCE,
    POSE_MODEL,
    as_point,
    best_club,
    body_anchor,
    golfer_index,
    outlined_text,
)

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pro_refs" / "Standard_10_frames"
OUTPUT_DIR = ROOT / "runs" / "standard_frames"
VIEWS = ("face_on", "down_the_line", "rear")
YAW_STEP = 15


def sheet_order(path: Path) -> int:
    match = re.match(r"P(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 999


def position_key(path: Path) -> str:
    match = re.match(r"P(\d+)", path.name, re.IGNORECASE)
    return f"p{int(match.group(1))}" if match else path.stem


def crop_panels(image: np.ndarray) -> list[np.ndarray]:
    """Three studio photos under the title, left to right."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 250
    columns = mask.mean(axis=0) > 0.05
    changes = np.diff(columns.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]
    spans = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start > 40]
    panels = []
    for x0, x1 in spans[:3]:
        rows = np.where(mask[:, x0:x1].mean(axis=1) > 0.05)[0]
        if len(rows) == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1])
        panels.append(image[y0 : y1 + 1, x0 : x1 + 1].copy())
    return panels


def point_or_none(value: np.ndarray | None) -> list[float] | None:
    if value is None or not np.isfinite(value).all():
        return None
    return [round(float(value[0]), 2), round(float(value[1]), 2)]


def pose_or_none(pose: np.ndarray | None) -> list[list[float]] | None:
    if pose is None or not np.isfinite(pose).all():
        return None
    return [[round(float(value), 5) for value in joint] for joint in pose]


def detect_view(crop: np.ndarray, pose_model: YOLO, club_model: YOLO) -> dict:
    points = np.full((len(GOLFPOSE_JOINTS), 2), np.nan)
    valid = np.zeros(len(GOLFPOSE_JOINTS), dtype=bool)
    box = np.full(4, np.nan)
    pose_result = pose_model.predict(crop, imgsz=640, verbose=False)[0]
    person = golfer_index(pose_result)
    anchor = None
    if person is not None:
        anchor = body_anchor(pose_result, person)
        xy = pose_result.keypoints.xy[person].cpu().numpy().astype(np.float64)
        if pose_result.keypoints.conf is not None:
            conf = pose_result.keypoints.conf[person].cpu().numpy()
        else:
            conf = np.ones(len(xy))
        points[:BODY_JOINTS], valid[:BODY_JOINTS] = coco_to_golfpose(xy, conf)
        if pose_result.boxes is not None and person < len(pose_result.boxes):
            box = pose_result.boxes.xyxy[person].cpu().numpy().astype(np.float64)

    diagonal = max(float(np.hypot(*crop.shape[:2])), 1.0)
    club_result = club_model.predict(
        crop, imgsz=640, conf=CLUB_CONFIDENCE, retina_masks=True, max_det=3, verbose=False
    )[0]
    detection = best_club(club_result, club_model.names, anchor, None, diagonal)
    grip = head = None
    if detection is not None:
        grip, head, _confidence, _face = detection
        if head is not None:
            points[JOINT["hosel"]] = head
            valid[JOINT["hosel"]] = True
        if grip is not None:
            points[JOINT["shaft"]] = grip
            valid[JOINT["shaft"]] = True
    return {"points": points, "valid": valid, "box": box, "grip": grip, "head": head, "found": person is not None}


def draw_labeled(crop: np.ndarray, detected: dict, view: str) -> np.ndarray:
    image = crop.copy()
    draw_skeleton_2d(image, detected["points"], detected["valid"])
    height, width = image.shape[:2]
    grip, head = detected["grip"], detected["head"]
    if grip is not None and head is not None and np.isfinite(grip).all() and np.isfinite(head).all():
        cv2.line(image, as_point(grip, width, height), as_point(head, width, height), (255, 255, 255), 2, cv2.LINE_AA)
    if head is not None and np.isfinite(head).all():
        cv2.circle(image, as_point(head, width, height), 5, (0, 0, 255), -1, cv2.LINE_AA)
    outlined_text(image, view.replace("_", " "), (12, 28), 0.7)
    return image


def lift_still(lifter, detected: dict, size: tuple[int, int]) -> np.ndarray | None:
    if not detected["found"] or not detected["valid"][:BODY_JOINTS].any():
        return None
    points = detected["points"][None]
    valid = detected["valid"][None]
    filled = fill_gaps(points, valid)
    boxes = detected["box"][None]
    swing = Swing2D(
        frames=1,
        fps=30.0,
        size=size,
        points=points,
        valid=valid,
        filled=filled,
        boxes=boxes,
        samples=[],
        wrist_y=np.zeros(1),
    )
    normalized, _params = golf_canvas(swing)
    return lift_sequence(lifter, normalized)[0]


def save_still(pose: np.ndarray | None, path: Path, title: str) -> None:
    if pose is None or not np.isfinite(pose).all():
        blank = np.full((360, 480, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "no pose", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), blank)
        return
    panel = Panel3D(480, 360, pose[None])
    image = panel.render(pose, np.zeros((0, 3)), np.zeros(0), show_club=True)
    cv2.putText(image, title, (12, image.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def unit(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8 or not np.isfinite(length):
        return None
    return vector / length


def bone_directions(pose: np.ndarray) -> dict[str, np.ndarray]:
    directions = {}
    for start_name, end_name in BONES:
        direction = unit(pose[JOINT[start_name]], pose[JOINT[end_name]])
        if direction is not None:
            directions[f"{start_name}-{end_name}"] = direction
    return directions


def mean_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = bone_directions(first)
    right = bone_directions(second)
    shared = [name for name in left if name in right]
    if not shared:
        return -1.0
    return float(np.mean([np.dot(left[name], right[name]) for name in shared]))


def normalize_pose(pose: np.ndarray) -> np.ndarray | None:
    if pose is None or not np.isfinite(pose).all():
        return None
    centered = pose - pose[JOINT["root"]]
    spine = float(np.linalg.norm(centered[JOINT["thorax"]]))
    if spine < 1e-6:
        return None
    return centered / spine


def mirror_pose(pose: np.ndarray) -> np.ndarray:
    mirrored = pose.copy()
    mirrored[:, 0] *= -1.0
    swapped = mirrored.copy()
    swapped[JOINTS_LEFT] = mirrored[JOINTS_RIGHT]
    swapped[JOINTS_RIGHT] = mirrored[JOINTS_LEFT]
    return swapped


def yaw_pose(pose: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    cosine, sine = float(np.cos(theta)), float(np.sin(theta))
    turned = pose.copy()
    x_coord, z_coord = pose[:, 0], pose[:, 2]
    turned[:, 0] = cosine * x_coord + sine * z_coord
    turned[:, 2] = -sine * x_coord + cosine * z_coord
    return turned


def align_to_anchor(pose: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, dict]:
    best_pose = pose
    best_score = -2.0
    best = {"mirrored": False, "yaw_deg": 0, "cosine": -1.0}
    for mirrored in (False, True):
        candidate = mirror_pose(pose) if mirrored else pose
        for degrees in range(0, 360, YAW_STEP):
            turned = yaw_pose(candidate, degrees)
            score = mean_cosine(turned, anchor)
            if score > best_score:
                best_score = score
                best_pose = turned
                best = {"mirrored": mirrored, "yaw_deg": degrees, "cosine": round(score, 4)}
    return best_pose, best


def fuse_views(poses: dict[str, np.ndarray | None]) -> dict:
    normalized = {name: normalize_pose(pose) for name, pose in poses.items()}
    available = {name: pose for name, pose in normalized.items() if pose is not None}
    if not available:
        return {"aligned": {}, "alignment": {}, "agreement": {}, "kept": [], "reference": None}
    anchor_name = "down_the_line" if "down_the_line" in available else next(iter(available))
    aligned = {anchor_name: available[anchor_name]}
    alignment = {anchor_name: {"mirrored": False, "yaw_deg": 0, "cosine": 1.0}}
    for name, pose in available.items():
        if name == anchor_name:
            continue
        aligned_pose, chosen = align_to_anchor(pose, available[anchor_name])
        aligned[name] = aligned_pose
        alignment[name] = chosen

    agreement = {}
    for name, pose in aligned.items():
        others = [mean_cosine(pose, other) for other_name, other in aligned.items() if other_name != name]
        agreement[name] = None if not others else round(float(np.mean(others)), 4)

    kept = [name for name in ("face_on", "down_the_line") if name in aligned]
    if not kept:
        kept = list(aligned)
    reference = None if not kept else np.mean([aligned[name] for name in kept], axis=0)
    return {
        "aligned": aligned,
        "alignment": alignment,
        "agreement": agreement,
        "kept": kept,
        "reference": reference,
    }


def joints_2d(detected: dict) -> dict[str, list[float] | None]:
    record = {}
    for name in GOLFPOSE_JOINTS:
        point = detected["points"][JOINT[name]]
        record[name] = point_or_none(point) if detected["valid"][JOINT[name]] else None
    return record


def process_sheet(path: Path, pose_model: YOLO, club_model: YOLO, lifter) -> None:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not read {path}")
    panels = crop_panels(image)
    if len(panels) != 3:
        raise SystemExit(f"{path.name} has {len(panels)} photo panels, expected 3")
    folder = OUTPUT_DIR / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    views = {}
    camera_poses = {}
    print(f"{path.name}", flush=True)
    for view, crop in zip(VIEWS, panels):
        detected = detect_view(crop, pose_model, club_model)
        cv2.imwrite(str(folder / f"{view}_labeled.jpg"), draw_labeled(crop, detected, view))
        pose = lift_still(lifter, detected, (crop.shape[1], crop.shape[0]))
        save_still(pose, folder / f"{view}_3d.jpg", view.replace("_", " "))
        camera_poses[view] = pose
        views[view] = {
            "joints_2d": joints_2d(detected),
            "grip_xy": point_or_none(None if detected["grip"] is None else np.asarray(detected["grip"])),
            "clubhead_xy": point_or_none(None if detected["head"] is None else np.asarray(detected["head"])),
            "pose_3d": pose_or_none(pose),
        }
        print(f"  {view}: pose {'yes' if pose is not None else 'no'}", flush=True)

    fused = fuse_views(camera_poses)
    save_still(fused["reference"], folder / "reference_3d.jpg", "reference")
    for view, record in views.items():
        record["aligned_3d"] = pose_or_none(fused["aligned"].get(view))
        record["alignment"] = fused["alignment"].get(view)
        record["agreement"] = fused["agreement"].get(view)
        record["kept"] = view in fused["kept"]
    summary = {
        "source": str(path.relative_to(ROOT)),
        "position": position_key(path),
        "joints": GOLFPOSE_JOINTS,
        "views": views,
        "reference_kept": fused["kept"],
        "reference_3d": pose_or_none(fused["reference"]),
    }
    (folder / "joints.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    arrays = {
        "joints": np.array(GOLFPOSE_JOINTS),
        "reference_3d": fused["reference"] if fused["reference"] is not None else np.full((len(GOLFPOSE_JOINTS), 3), np.nan),
    }
    for view in VIEWS:
        pose = camera_poses[view]
        arrays[f"{view}_3d"] = pose if pose is not None else np.full((len(GOLFPOSE_JOINTS), 3), np.nan)
        aligned = fused["aligned"].get(view)
        arrays[f"{view}_aligned"] = aligned if aligned is not None else np.full((len(GOLFPOSE_JOINTS), 3), np.nan)
    np.savez(folder / "joints.npz", **arrays)


def main() -> None:
    sheets = sorted(INPUT_DIR.glob("*.jpg"), key=sheet_order)
    if not sheets:
        raise SystemExit(f"No sheets in {INPUT_DIR}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Trained golf-club weights not found: {MODEL_PATH}")
    pose_model = YOLO(POSE_MODEL)
    club_model = YOLO(str(MODEL_PATH))
    lifter = load_lifter(CHECKPOINT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sheets:
        process_sheet(path, pose_model, club_model, lifter)
    print(f"Saved: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
