"""Score a lifted user swing against the ten standard reference skeletons.

The user folder comes from golf_pose_3d.py. Each standard pose is reference_3d
in runs/standard_frames. The score compares relative joint angles, so a
difference in camera heading does not change it.

Usage:
    python -m golf_workflows.compare_standard_frames runs/golf_pose_3d/<stem>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from golf_workflows.golf_pose_3d import JOINT, OUTPUT_DIR
from golf_workflows.score_swing import POSITIONS, angle_score, frame_of, lead_of, load_swing, unit
from golf_workflows.standard_frames import mean_cosine, normalize_pose, save_still, yaw_pose

ROOT = Path(__file__).resolve().parents[1]
STANDARD_DIR = ROOT / "runs" / "standard_frames"
YAW_STEP = 15
UP = np.array([0.0, -1.0, 0.0])
POSTURE = (
    "Alignment",
    "Hip Bend",
    "Hip Depth",
    "Knee Bend",
    "Hands",
    "Stance Width",
    "Flying Elbow",
    "Trail Knee",
    "Bend",
    "Shaft Lean",
    "Spine Tilt",
    "Lead Shoulder",
    "Lead Hip",
    "Lead Knee",
)


def standard_path(key: str, standard_dir: Path | None = None) -> Path | None:
    matches = sorted((standard_dir or STANDARD_DIR).glob(f"{key.upper()}-*"))
    folders = [path for path in matches if (path / "joints.npz").is_file() and (path / "reference_3d.jpg").is_file()]
    return folders[0] if folders else None


def best_yaw(user_pose: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    best_pose = user_pose
    best_score = -2.0
    best_yaw = 0
    for degrees in range(0, 360, YAW_STEP):
        turned = yaw_pose(user_pose, degrees)
        score = mean_cosine(turned, reference)
        if score > best_score:
            best_score = score
            best_pose = turned
            best_yaw = degrees
    return best_pose, best_yaw


def point(pose: np.ndarray, name: str) -> np.ndarray | None:
    value = pose[JOINT[name]]
    if not np.isfinite(value).all():
        return None
    return value


def direction(pose: np.ndarray, start_name: str, end_name: str) -> np.ndarray | None:
    start = point(pose, start_name)
    end = point(pose, end_name)
    if start is None or end is None:
        return None
    return unit(start, end)


def opening(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def joint_angle(pose: np.ndarray, proximal: str, joint: str, distal: str) -> float | None:
    center = point(pose, joint)
    proximal_point = point(pose, proximal)
    distal_point = point(pose, distal)
    if center is None or proximal_point is None or distal_point is None:
        return None
    return opening(unit(center, proximal_point), unit(center, distal_point))


def horizontal(vector: np.ndarray) -> np.ndarray | None:
    flat = np.array([vector[0], 0.0, vector[2]], dtype=np.float64)
    length = float(np.linalg.norm(flat))
    if length < 1e-8 or not np.isfinite(length):
        return None
    return flat / length


def sides(lead: str) -> tuple[str, str]:
    if lead == "right":
        return "right", "left"
    return "left", "right"


def average_thigh(pose: np.ndarray) -> np.ndarray | None:
    thighs = [direction(pose, f"{side}_hip", f"{side}_knee") for side in ("left", "right")]
    if any(thigh is None for thigh in thighs):
        return None
    total = thighs[0] + thighs[1]
    length = float(np.linalg.norm(total))
    if length < 1e-8:
        return None
    return total / length


def shoulder_axes(pose: np.ndarray, lead: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    lead_side, trail_side = sides(lead)
    lead_point = point(pose, f"{lead_side}_shoulder")
    trail_point = point(pose, f"{trail_side}_shoulder")
    if lead_point is None or trail_point is None:
        return None, None
    toward_lead = horizontal(lead_point - trail_point)
    if toward_lead is None:
        return None, None
    forward = np.cross(UP, toward_lead)
    length = float(np.linalg.norm(forward))
    if length < 1e-8:
        return toward_lead, None
    return toward_lead, forward / length


def signed_from_up(vector: np.ndarray | None, toward: np.ndarray | None) -> float | None:
    if vector is None or toward is None:
        return None
    return float(np.degrees(np.arctan2(np.dot(vector, toward), np.dot(vector, UP))))


def posture_gap(user_angle: float, reference_angle: float) -> float:
    gap = abs(user_angle - reference_angle) % 360.0
    if gap > 180.0:
        gap = 360.0 - gap
    return angle_score(0.0, gap)


def posture_angles(pose: np.ndarray, lead: str) -> dict[str, float | None]:
    lead_side, trail_side = sides(lead)
    spine = direction(pose, "root", "thorax")
    thigh = average_thigh(pose)
    toward_lead, forward = shoulder_axes(pose, lead)
    left_shoulder = point(pose, "left_shoulder")
    right_shoulder = point(pose, "right_shoulder")
    left_hip = point(pose, "left_hip")
    right_hip = point(pose, "right_hip")
    shoulder_line = None if left_shoulder is None or right_shoulder is None else horizontal(left_shoulder - right_shoulder)
    hip_line = None if left_hip is None or right_hip is None else horizontal(left_hip - right_hip)
    wrists = [point(pose, "left_wrist"), point(pose, "right_wrist")]
    thorax = point(pose, "thorax")
    hands = None
    if spine is not None and thorax is not None and all(wrist is not None for wrist in wrists):
        hands = opening(spine, unit(thorax, (wrists[0] + wrists[1]) / 2.0))
    knees = [
        joint_angle(pose, f"{side}_hip", f"{side}_knee", f"{side}_foot") for side in ("left", "right")
    ]
    knee_bend = None if any(angle is None for angle in knees) else float(np.mean(knees))
    if shoulder_line is None or hip_line is None:
        alignment = None
    else:
        alignment = float(np.degrees(np.arccos(float(np.clip(abs(np.dot(shoulder_line, hip_line)), -1.0, 1.0)))))
    return {
        "Alignment": alignment,
        "Hip Bend": opening(spine, thigh),
        "Hip Depth": opening(thigh, UP),
        "Knee Bend": knee_bend,
        "Hands": hands,
        "Stance Width": joint_angle(pose, "left_foot", "root", "right_foot"),
        "Flying Elbow": opening(
            direction(pose, f"{trail_side}_shoulder", f"{trail_side}_elbow"),
            direction(pose, f"{trail_side}_shoulder", f"{trail_side}_hip"),
        ),
        "Trail Knee": joint_angle(pose, f"{trail_side}_hip", f"{trail_side}_knee", f"{trail_side}_foot"),
        "Bend": signed_from_up(spine, forward),
        "Shaft Lean": signed_from_up(direction(pose, "shaft", "hosel"), toward_lead),
        "Spine Tilt": signed_from_up(spine, None if toward_lead is None else -toward_lead),
        "Lead Shoulder": opening(
            direction(pose, "thorax", f"{lead_side}_shoulder"),
            direction(pose, f"{lead_side}_shoulder", f"{lead_side}_elbow"),
        ),
        "Lead Hip": opening(
            direction(pose, f"{lead_side}_hip", "root"),
            direction(pose, f"{lead_side}_hip", f"{lead_side}_knee"),
        ),
        "Lead Knee": joint_angle(pose, f"{lead_side}_hip", f"{lead_side}_knee", f"{lead_side}_foot"),
    }


def side_by_side(user_image: Path, reference_image: Path, destination: Path) -> None:
    left = cv2.imread(str(user_image))
    right = cv2.imread(str(reference_image))
    if left is None or right is None:
        return
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
    cv2.imwrite(str(destination), np.hstack([left, right]))


def compare(user_folder: Path, standard_dir: Path | None = None,
            output_dir: Path | None = None, selected_swing: dict | None = None) -> dict:
    swing = selected_swing if selected_swing is not None else load_swing(user_folder)
    if swing is None:
        raise SystemExit(f"No *_3d.npz in {user_folder}")
    lead = lead_of(swing["summary"])
    output = output_dir or user_folder / "standard_compare"
    output.mkdir(parents=True, exist_ok=True)
    positions = {}
    parameter_scores = []
    frames = {}
    for key in POSITIONS:
        index = frame_of(swing["summary"], key)
        frame = swing["summary"].get("source_frame_indices", {}).get(key, index)
        frames[key] = frame
        folder = standard_path(key, standard_dir)
        user_pose = None if index is None or not 0 <= index < len(swing["poses"]) else swing["poses"][index]
        if folder is None or user_pose is None or not np.isfinite(user_pose).all():
            positions[key] = None
            continue
        reference = np.load(folder / "joints.npz")["reference_3d"]
        user_norm = normalize_pose(user_pose)
        reference_norm = normalize_pose(reference)
        if user_norm is None or reference_norm is None:
            positions[key] = None
            continue
        aligned, yaw = best_yaw(user_norm, reference_norm)
        label = key.upper()
        user_still = output / f"{label}_user_3d.jpg"
        save_still(aligned, user_still, label)
        side_by_side(user_still, folder / "reference_3d.jpg", output / f"{label}_vs_reference.jpg")
        user_angles = posture_angles(user_norm, lead)
        reference_angles = posture_angles(reference_norm, lead)
        details = {}
        values = []
        for name in POSTURE:
            user_angle = user_angles[name]
            reference_angle = reference_angles[name]
            if user_angle is None or reference_angle is None:
                continue
            score = posture_gap(user_angle, reference_angle)
            details[name] = score
            values.append(score)
            parameter_scores.append(score)
        positions[key] = None if not values else {
            "frame": frame,
            "yaw_deg": yaw,
            "standard": folder.name,
            "score": round(float(np.mean(values)), 1),
            "parameters": details,
        }
        print(f"{label} frame {frame} score {None if positions[key] is None else positions[key]['score']}", flush=True)
    total = None if not parameter_scores else round(float(np.mean(parameter_scores)), 1)
    result = {
        "pose_score": total,
        "weight": "equal",
        "frames": frames,
        "positions": positions,
    }
    destination = output / "score.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Pose score: {total}")
    print(f"Score: {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a user swing against the ten standard 3D poses.")
    parser.add_argument("user", nargs="?", type=Path, help="golf_pose_3d output folder.")
    args = parser.parse_args()
    folder = args.user or (OUTPUT_DIR / "1785648274128019000_swing")
    if not folder.is_dir():
        raise SystemExit(f"User swing folder not found: {folder}")
    compare(folder)


if __name__ == "__main__":
    main()
