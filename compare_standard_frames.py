"""Score a lifted user swing against the ten standard reference skeletons.

The user folder comes from golf_pose_3d.py. Each standard pose is reference_3d
in runs/standard_frames.

Usage:
    python compare_standard_frames.py runs/golf_pose_3d/<stem>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from golf_pose_3d import OUTPUT_DIR
from score_swing import (
    POSITIONS,
    angle_score,
    criteria,
    direction_score,
    frame_of,
    joint_direction,
    lead_of,
    limbs,
    load_swing,
)
from standard_frames import mean_cosine, normalize_pose, save_still, yaw_pose
from swing_phases import wrist_shaft_degrees

ROOT = Path(__file__).resolve().parent
STANDARD_DIR = ROOT / "runs" / "standard_frames"
YAW_STEP = 15


def standard_path(key: str) -> Path | None:
    matches = sorted(STANDARD_DIR.glob(f"{key.upper()}-*"))
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


def pair_names(spec: tuple, lead: str) -> tuple[str, str] | None:
    if spec[0] == "face":
        return None
    if spec[0] == "limb":
        return limbs(lead)[spec[2]]
    return spec[2], spec[3]


def parameter_score(user_pose: np.ndarray, reference: np.ndarray, spec: tuple, lead: str) -> float | None:
    if spec[0] == "wrist":
        user_angle = wrist_shaft_degrees(user_pose[None], 0, lead)
        reference_angle = wrist_shaft_degrees(reference[None], 0, lead)
        if user_angle is None or reference_angle is None:
            return None
        return angle_score(user_angle, reference_angle)
    names = pair_names(spec, lead)
    if names is None:
        return None
    user_direction = joint_direction(user_pose, *names)
    reference_direction = joint_direction(reference, *names)
    if user_direction is None or reference_direction is None:
        return None
    return direction_score(float(np.dot(user_direction, reference_direction)))


def side_by_side(user_image: Path, reference_image: Path, destination: Path) -> None:
    left = cv2.imread(str(user_image))
    right = cv2.imread(str(reference_image))
    if left is None or right is None:
        return
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
    cv2.imwrite(str(destination), np.hstack([left, right]))


def compare(user_folder: Path) -> dict:
    swing = load_swing(user_folder)
    if swing is None:
        raise SystemExit(f"No *_3d.npz in {user_folder}")
    lead = lead_of(swing["summary"])
    checklist = criteria()
    output = user_folder / "standard_compare"
    output.mkdir(parents=True, exist_ok=True)
    positions = {}
    parameter_scores = []
    frames = {}
    for key in POSITIONS:
        frame = frame_of(swing["summary"], key)
        frames[key] = frame
        folder = standard_path(key)
        user_pose = None if frame is None else swing["poses"][frame]
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
        details = {}
        values = []
        for spec in checklist[key]:
            score = parameter_score(aligned, reference_norm, spec, lead)
            if score is None:
                continue
            details[spec[1]] = score
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
