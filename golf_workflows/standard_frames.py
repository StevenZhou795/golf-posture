"""Lift the ten standard position sheets into one 3D reference pose each.

Each sheet shows the same position from face-on, down-the-line, and rear.
The face-on lift is the 3D space. The down-the-line view supplies the side axis.

Usage:
    python -m golf_workflows.standard_frames
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from golf_workflows.golf_pose_3d import (
    BONES,
    BODY_JOINTS,
    GOLFPOSE_JOINTS,
    JOINT,
    JOINTS_LEFT,
    JOINTS_RIGHT,
    CHECKPOINT,
    CHECKPOINT_22,
    EXTRA_CLUB_JOINTS,
    Panel3D,
    Swing2D,
    coco_to_golfpose,
    draw_skeleton_2d,
    fill_gaps,
    golf_canvas,
    lift_sequence,
    load_lifter,
    joint_names,
    pad_keypoints,
)
from predict_seg import MODEL_PATH
from golf_workflows.trace_swing_plane import (
    CLUB_CONFIDENCE,
    POSE_MODEL,
    as_point,
    best_club,
    best_club_box,
    body_anchor,
    golfer_index,
    outlined_text,
)

ROOT = Path(__file__).resolve().parents[1]
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


def detect_view(crop: np.ndarray, pose_model: YOLO, club_model: YOLO | None, club_pose_model=None,
                club_crop_model=None) -> dict:
    count = 22 if club_pose_model is not None else len(GOLFPOSE_JOINTS)
    points = np.full((count, 2), np.nan)
    valid = np.zeros(count, dtype=bool)
    club_confidence = np.zeros(5, dtype=np.float32)
    club_pose_details = None
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
    crop_details = None
    if club_crop_model is not None:
        detection = None
        official_club_box, crop_details = club_crop_model.club_box(crop)
    else:
        club_result = club_model.predict(
            crop, imgsz=640, conf=CLUB_CONFIDENCE, retina_masks=True, max_det=3, verbose=False
        )[0]
        detection = best_club(club_result, club_model.names, anchor, None, diagonal,
                              return_box=club_pose_model is not None)
        official_club_box = None
    grip = head = None
    if club_pose_model is not None:
        club_box = official_club_box
        golfer_box_fallback = False
        if club_crop_model is None and detection is not None:
            _seg_grip, _seg_head, _confidence, _face, club_box = detection
        elif club_crop_model is None:
            club_box = best_club_box(club_result, club_model.names, anchor, diagonal)
            if club_box is None and np.isfinite(box).all():
                club_box = box
                golfer_box_fallback = True
        if club_box is not None:
            inference = (club_pose_model.predict_coarse_to_fine if golfer_box_fallback
                         else club_pose_model.predict)
            landmarks, club_confidence, club_pose_details = inference(crop, club_box)
            if crop_details is not None:
                club_pose_details["crop_detector"] = crop_details
            good = club_confidence >= club_pose_model.confidence_threshold
            points[17:22] = landmarks
            valid[17:22] = good
            grip = landmarks[0] if good[0] else None
            head = landmarks[1] if good[1] else None
    elif detection is not None:
        grip, head, confidence, _face = detection
        club_confidence[:2] = confidence
        if head is not None:
            points[JOINT["hosel"]] = head
            valid[JOINT["hosel"]] = True
        if grip is not None:
            points[JOINT["shaft"]] = grip
            valid[JOINT["shaft"]] = True
    return {"points": points, "valid": valid, "box": box, "grip": grip, "head": head, "found": person is not None,
            "pose_backend_details": getattr(pose_result, "vitpose_details", None),
            "club_confidence": club_confidence, "club_pose_details": club_pose_details,
            "club_crop_details": crop_details}


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


def save_still(pose: np.ndarray | None, path: Path, title: str, *, show_club: bool = True) -> None:
    if pose is None or not np.isfinite(pose).all():
        blank = np.full((360, 480, 3), 255, dtype=np.uint8)
        cv2.putText(blank, "no pose", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), blank)
        return
    panel = Panel3D(480, 360, pose[None])
    image = panel.render(pose, np.zeros((0, 3)), np.zeros(0), show_club=show_club)
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


def camera_side_axis(mirrored: bool, yaw_deg: float) -> np.ndarray:
    """Down-the-line image horizontal, mapped into the face-on camera."""
    axis = np.array([[-1.0 if mirrored else 1.0, 0.0, 0.0]])
    turned = yaw_pose(axis, yaw_deg)[0]
    length = float(np.linalg.norm(turned))
    if length < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return turned / length


def replace_side(base: np.ndarray, donor: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Keep the base pose and copy only the component along axis from the donor."""
    base_part = base @ axis
    donor_part = donor @ axis
    return base + (donor_part - base_part)[:, None] * axis


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
        return {
            "aligned": {},
            "alignment": {},
            "agreement": {},
            "kept": [],
            "reference": None,
            "side_axis": None,
        }
    anchor_name = "face_on" if "face_on" in available else next(iter(available))
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
    side = None
    reference = aligned.get("face_on")
    if reference is not None and "down_the_line" in aligned and anchor_name == "face_on":
        chosen = alignment["down_the_line"]
        side = camera_side_axis(chosen["mirrored"], chosen["yaw_deg"])
        chosen["side_axis"] = [round(float(value), 5) for value in side]
        reference = replace_side(aligned["face_on"], aligned["down_the_line"], side)
    elif reference is None:
        reference = aligned.get("down_the_line")
    return {
        "aligned": aligned,
        "alignment": alignment,
        "agreement": agreement,
        "kept": kept,
        "reference": reference,
        "side_axis": side,
    }


def build_reference_club_overlay(fused: dict, detections: dict) -> tuple[np.ndarray | None, dict]:
    """Build a render-only club overlay from one complete, reliable GolfPose view.

    The five official channels retain their model order and their aligned MixSTE
    coordinates.  This function never changes ``fused['reference']``.
    """
    reference = fused.get("reference")
    if reference is None or len(reference) != 22:
        return None, {"applied": False, "reason": "22-joint reference unavailable"}
    candidates = []
    for view, detected in detections.items():
        confidence = np.asarray(detected["club_confidence"], dtype=float)
        points = np.asarray(detected["points"], dtype=float)
        valid = np.asarray(detected["valid"], dtype=bool)
        donor = fused.get("aligned", {}).get(view)
        if (len(points) != 22 or len(valid) != 22 or donor is None
                or not valid[17:22].all() or not np.isfinite(points[17:22]).all()
                or not np.isfinite(donor[17:22]).all()):
            continue
        shaft_px = float(np.linalg.norm(points[18] - points[17]))
        if shaft_px < 12:
            continue
        details = detected.get("club_pose_details") or {}
        fallback_penalty = .05 if details.get("crop_fallback") else 0.0
        score = float(np.min(confidence)) + .25 * float(np.mean(np.clip(confidence, 0, 1))) - fallback_penalty
        candidates.append((score, view, shaft_px, confidence))
    if not candidates:
        return None, {
            "applied": False,
            "reason": "no view has all five reliable GolfPose club landmarks",
            "numerical_reference_modified": False,
        }
    score, view, shaft_px, confidence = max(candidates, key=lambda item: item[0])
    donor = fused["aligned"][view]
    overlay = reference.copy()
    overlay[17:22] = donor[17:22]
    direction_2d = detections[view]["points"][18] - detections[view]["points"][17]
    direction_2d /= max(float(np.linalg.norm(direction_2d)), 1e-8)
    return overlay, {
        "applied": True,
        "method": "direct_aligned_22_keypoint_view_render_only",
        "source_view": view,
        "source_score": round(float(score), 4),
        "source_confidence": [round(float(value), 5) for value in confidence],
        "source_shaft_length_px": round(float(shaft_px), 2),
        "source_shaft_direction_2d": [round(float(value), 5) for value in direction_2d],
        "landmark_order": ["shaft", "hosel", "heel", "toe_down", "toe_up"],
        "numerical_reference_modified": False,
        "caveat": "Visualization overlay only; not used as a numerical reference or scoring input",
    }


def joints_2d(detected: dict) -> dict[str, list[float] | None]:
    record = {}
    for index, name in enumerate(joint_names(detected["points"].shape[0])):
        point = detected["points"][index]
        record[name] = point_or_none(point) if detected["valid"][index] else None
    return record


def process_sheet(path: Path, pose_model: YOLO, club_model: YOLO, lifter, club_pose_model=None,
                  club_crop_model=None) -> dict:
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
    names = joint_names(lifter.Spatial_pos_embed.shape[1])
    inputs = {}
    detections = {}
    print(f"{path.name}", flush=True)
    for view, crop in zip(VIEWS, panels):
        detected = detect_view(crop, pose_model, club_model, club_pose_model, club_crop_model)
        detected["points"], detected["valid"] = pad_keypoints(detected["points"], detected["valid"], len(names))
        filled = fill_gaps(detected["points"][None], detected["valid"][None])[0]
        inputs[f"{view}_2d"] = detected["points"]
        inputs[f"{view}_2d_valid"] = detected["valid"]
        inputs[f"{view}_2d_filled"] = filled
        inputs[f"{view}_club_confidence"] = detected["club_confidence"]
        detections[view] = detected
        cv2.imwrite(str(folder / f"{view}_labeled.jpg"), draw_labeled(crop, detected, view))
        pose = lift_still(lifter, detected, (crop.shape[1], crop.shape[0]))
        save_still(pose, folder / f"{view}_3d.jpg", view.replace("_", " "))
        camera_poses[view] = pose
        views[view] = {
            "joints_2d": joints_2d(detected),
            "pose_backend_details": detected["pose_backend_details"],
            "club_pose_details": detected["club_pose_details"],
            "club_crop_details": detected["club_crop_details"],
            "club_landmark_confidence": detected["club_confidence"].tolist(),
            "imputed_input_joints": [name for index, name in enumerate(names) if not detected["valid"][index]],
            "extra_club_input_fill": "hosel" if len(names) == 22 and club_pose_model is None else None,
            "grip_xy": point_or_none(None if detected["grip"] is None else np.asarray(detected["grip"])),
            "clubhead_xy": point_or_none(None if detected["head"] is None else np.asarray(detected["head"])),
            "pose_3d": pose_or_none(pose),
        }
        print(f"  {view}: pose {'yes' if pose is not None else 'no'}", flush=True)

    fused = fuse_views(camera_poses)
    club_overlay_pose, club_overlay = build_reference_club_overlay(fused, detections)
    save_still(fused["reference"], folder / "reference_3d.jpg", "reference (body)", show_club=False)
    club_overlay_path = folder / "reference_3d_club_overlay.jpg"
    if club_overlay_pose is not None:
        save_still(club_overlay_pose, club_overlay_path, "reference (club overlay)")
    else:
        club_overlay_path.unlink(missing_ok=True)
    for view, record in views.items():
        record["aligned_3d"] = pose_or_none(fused["aligned"].get(view))
        record["alignment"] = fused["alignment"].get(view)
        record["agreement"] = fused["agreement"].get(view)
        record["kept"] = view in fused["kept"]
    summary = {
        "source": str(path.relative_to(ROOT)),
        "position": position_key(path),
        "joints": names,
        "models": {"pose": getattr(pose_model,"metadata",{}).get("pose_checkpoint",POSE_MODEL),
                   "pose_backend": getattr(pose_model,"metadata",{}),
                   "club": str(MODEL_PATH.relative_to(ROOT)) if club_model is not None else None,
                   "club_pose_backend": (getattr(club_pose_model, "metadata", None)
                                         if club_pose_model is not None else {"club_pose_backend": "segmentation"}),
                   "club_crop_backend": (getattr(club_crop_model, "metadata", None)
                                         if club_crop_model is not None else {"club_crop_backend": "segmentation"}),
                   "lifter": str(lifter.checkpoint_path.relative_to(ROOT))},
        "experimental": len(names) == 22,
        "unmeasured_extra_club_inputs": EXTRA_CLUB_JOINTS if len(names) == 22 and club_pose_model is None else [],
        "views": views,
        "reference_kept": fused["kept"],
        "side_axis": None if fused["side_axis"] is None else [round(float(value), 5) for value in fused["side_axis"]],
        "reference_3d": pose_or_none(fused["reference"]),
        "reference_club_overlay": club_overlay,
    }
    (folder / "joints.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    arrays = {
        "joints": np.array(names),
        "reference_3d": fused["reference"] if fused["reference"] is not None else np.full((len(names), 3), np.nan),
        "reference_3d_club_overlay": (club_overlay_pose
            if club_overlay_pose is not None else np.full((len(names), 3), np.nan)),
        **inputs,
    }
    for view in VIEWS:
        pose = camera_poses[view]
        arrays[f"{view}_3d"] = pose if pose is not None else np.full((len(names), 3), np.nan)
        aligned = fused["aligned"].get(view)
        arrays[f"{view}_aligned"] = aligned if aligned is not None else np.full((len(names), 3), np.nan)
    np.savez(folder / "joints.npz", **arrays)
    return summary


def _finite_pose(pose: np.ndarray) -> np.ndarray | None:
    if pose is None or not np.isfinite(pose).all():
        return None
    return np.asarray(pose, dtype=np.float64)


def rebuild_references() -> None:
    """Rewrite reference_3d from the camera poses already stored in joints.npz."""
    folders = sorted((path for path in OUTPUT_DIR.iterdir() if path.is_dir()), key=lambda path: sheet_order(path))
    if not folders:
        raise SystemExit(f"No position folders in {OUTPUT_DIR}")
    for folder in folders:
        npz_path = folder / "joints.npz"
        json_path = folder / "joints.json"
        if not npz_path.is_file() or not json_path.is_file():
            raise SystemExit(f"Missing joints files in {folder}")
        stored = np.load(npz_path)
        arrays = {key: stored[key] for key in stored.files}
        stored.close()
        count = len(arrays["joints"])
        poses = {view: _finite_pose(arrays[f"{view}_3d"]) for view in VIEWS}
        fused = fuse_views(poses)
        save_still(fused["reference"], folder / "reference_3d.jpg", "reference")
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        for view, record in summary["views"].items():
            record["aligned_3d"] = pose_or_none(fused["aligned"].get(view))
            record["alignment"] = fused["alignment"].get(view)
            record["agreement"] = fused["agreement"].get(view)
            record["kept"] = view in fused["kept"]
        summary["reference_kept"] = fused["kept"]
        summary["side_axis"] = (
            None if fused["side_axis"] is None else [round(float(value), 5) for value in fused["side_axis"]]
        )
        summary["reference_3d"] = pose_or_none(fused["reference"])
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        arrays["reference_3d"] = (
            fused["reference"] if fused["reference"] is not None else np.full((count, 3), np.nan)
        )
        for view in VIEWS:
            aligned = fused["aligned"].get(view)
            arrays[f"{view}_aligned"] = aligned if aligned is not None else np.full((count, 3), np.nan)
        np.savez(npz_path, **arrays)
        face = fused["aligned"].get("face_on")
        donor = fused["aligned"].get("down_the_line")
        reference = fused["reference"]
        side = fused["side_axis"]
        if face is not None and donor is not None and reference is not None and side is not None:
            kept = reference - np.outer((reference - face) @ side, side)
            copied = float(np.max(np.abs((reference - donor) @ side)))
            picture = float(np.max(np.abs(kept - face)))
            print(
                f"{folder.name}: yaw {fused['alignment']['down_the_line']['yaw_deg']} "
                f"mirror {fused['alignment']['down_the_line']['mirrored']} "
                f"side {[round(float(value), 3) for value in side]} "
                f"picture_err {picture:.2e} side_err {copied:.2e}",
                flush=True,
            )
        else:
            print(f"{folder.name}: reference without a side axis", flush=True)
    print(f"Saved: {OUTPUT_DIR}", flush=True)


def main(argv: list[str] | None = None) -> None:
    global INPUT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lifter", choices=("17+2", "17+5"), default="17+2")
    parser.add_argument("--pose-backend", choices=("yolo26n-pose", "vitpose-huge"), default="yolo26n-pose")
    parser.add_argument("--pose-cpu-threads", type=int, default=4)
    parser.add_argument("--club-pose-backend", choices=("segmentation", "vitpose-huge"), default="segmentation")
    parser.add_argument("--club-crop-backend", choices=("segmentation", "official-yolox-2cls"), default="segmentation")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.club_pose_backend == "vitpose-huge" and args.lifter != "17+5":
        parser.error("--club-pose-backend vitpose-huge requires --lifter 17+5")
    if args.club_crop_backend == "official-yolox-2cls" and args.club_pose_backend != "vitpose-huge":
        parser.error("--club-crop-backend official-yolox-2cls requires --club-pose-backend vitpose-huge")
    INPUT_DIR = args.input_dir.resolve()
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir.resolve()
    elif args.pose_backend == "vitpose-huge":
        OUTPUT_DIR = ROOT / "runs" / ("standard_frames_vitpose_huge_" + args.lifter.replace("+", "plus"))
    elif args.lifter == "17+5":
        OUTPUT_DIR = ROOT / "runs" / "standard_frames_17plus5"
    if args.rebuild:
        rebuild_references()
        return
    sheets = sorted(INPUT_DIR.glob("*.jpg"), key=sheet_order)
    if not sheets:
        raise SystemExit(f"No sheets in {INPUT_DIR}")
    needs_segmentation = args.club_crop_backend == "segmentation" or args.club_pose_backend == "segmentation"
    if needs_segmentation and not MODEL_PATH.is_file():
        raise SystemExit(f"Trained golf-club weights not found: {MODEL_PATH}")
    from golf_workflows.vitpose_tracking import load_pose_backend
    pose_model = load_pose_backend(args.pose_backend, cpu_threads=args.pose_cpu_threads)
    club_model = YOLO(str(MODEL_PATH)) if needs_segmentation else None
    from golf_workflows.club_vitpose import load_club_pose_backend
    try:
        club_pose_model = load_club_pose_backend(args.club_pose_backend)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    from golf_workflows.official_yolox import load_club_crop_backend
    try:
        club_crop_model = load_club_crop_backend(args.club_crop_backend)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    lifter = load_lifter(CHECKPOINT_22 if args.lifter == "17+5" else CHECKPOINT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = [process_sheet(path, pose_model, club_model, lifter, club_pose_model, club_crop_model)
               for path in sheets]
    report = {
        "pose_backend": args.pose_backend, "models": reports[0]["models"],
        "lifter": args.lifter, "joint_count": len(reports[0]["joints"]),
        "input_dir": str(INPUT_DIR), "output_dir": str(OUTPUT_DIR),
        "sheets": len(reports), "views": len(reports) * 3,
        "finite_camera_poses": sum(v["pose_3d"] is not None for r in reports for v in r["views"].values()),
        "reference_poses": sum(r["reference_3d"] is not None for r in reports),
        "reference_club_overlays": sum(r["reference_club_overlay"]["applied"] for r in reports),
        "experimental": args.lifter == "17+5",
        "unmeasured_extra_club_inputs": (EXTRA_CLUB_JOINTS
                                           if args.lifter == "17+5" and club_pose_model is None else []),
        "extra_club_input_fill": ("hosel" if args.lifter == "17+5" and club_pose_model is None else None),
    }
    if club_pose_model is not None:
        confidence = np.asarray(
            [view["club_landmark_confidence"] for item in reports for view in item["views"].values()]
        )
        report["club_landmarks"] = ["shaft", "hosel", "heel", "toe_down", "toe_up"]
        report["club_landmark_valid_views"] = (
            confidence >= club_pose_model.confidence_threshold
        ).sum(axis=0).tolist()
    if club_crop_model is not None:
        crop_records = [view.get("club_crop_details") for item in reports for view in item["views"].values()]
        report["club_crop_detected_views"] = sum(
            bool(record and record.get("selected_club_bbox_xyxy") is not None) for record in crop_records
        )
        report["club_crop_total_views"] = len(crop_records)
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
