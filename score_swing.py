"""Score one lifted swing against a folder of lifted standard swings.

Both sides come from golf_pose_3d.py. The pose score compares the ten
HackMotion positions. Every parameter that both swings can supply has the
same weight. The clubhead path is reported separately.

Usage:
    python score_swing.py --standards runs/golf_pose_3d/standards --user runs/golf_pose_3d/<stem>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from golf_pose_3d import JOINT
from swing_phases import wrist_shaft_degrees

PATH_SAMPLES = 50
POSITIONS = [f"p{index}" for index in range(1, 11)]


def unit(start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1e-8 or not np.isfinite(length):
        return None
    return vector / length


def lead_of(summary: dict) -> str:
    lead = summary.get("lead_side")
    if lead in ("left", "right"):
        return lead
    return "left" if summary.get("handedness", "right") == "right" else "right"


def limbs(lead: str) -> dict[str, tuple[str, str]]:
    if lead == "right":
        lead_side, trail_side = "right", "left"
    else:
        lead_side, trail_side = "left", "right"
    return {
        "lead_upper": (f"{lead_side}_shoulder", f"{lead_side}_elbow"),
        "trail_upper": (f"{trail_side}_shoulder", f"{trail_side}_elbow"),
        "lead_forearm": (f"{lead_side}_elbow", f"{lead_side}_wrist"),
        "trail_forearm": (f"{trail_side}_elbow", f"{trail_side}_wrist"),
        "lead_thigh": (f"{lead_side}_hip", f"{lead_side}_knee"),
        "trail_thigh": (f"{trail_side}_hip", f"{trail_side}_knee"),
        "lead_shin": (f"{lead_side}_knee", f"{lead_side}_foot"),
        "trail_shin": (f"{trail_side}_knee", f"{trail_side}_foot"),
        "lead_to_trail_wrist": (f"{lead_side}_wrist", f"{trail_side}_wrist"),
        "head_to_lead_wrist": ("head", f"{lead_side}_wrist"),
    }


def criteria() -> dict[str, list[tuple]]:
    """Parameters for each HackMotion position. Each tuple is one equal-weight item."""
    spine = ("joints", "spine", "root", "thorax")
    shoulders = ("joints", "shoulders", "left_shoulder", "right_shoulder")
    hips = ("joints", "hips", "left_hip", "right_hip")
    shaft = ("joints", "shaft", "shaft", "hosel")
    lead_upper = ("limb", "lead_upper", "lead_upper")
    trail_upper = ("limb", "trail_upper", "trail_upper")
    lead_forearm = ("limb", "lead_forearm", "lead_forearm")
    trail_forearm = ("limb", "trail_forearm", "trail_forearm")
    lead_thigh = ("limb", "lead_thigh", "lead_thigh")
    trail_thigh = ("limb", "trail_thigh", "trail_thigh")
    lead_shin = ("limb", "lead_shin", "lead_shin")
    trail_shin = ("limb", "trail_shin", "trail_shin")
    wrist_cross = ("limb", "lead_to_trail_wrist", "lead_to_trail_wrist")
    hands_by_head = ("limb", "head_to_lead_wrist", "head_to_lead_wrist")
    wrist = ("wrist", "lead_wrist_shaft")
    return {
        "p1": [spine, shoulders, hips, lead_thigh, trail_thigh, lead_shin, trail_shin, shaft],
        "p2": [shaft, lead_upper, trail_upper, lead_forearm, trail_forearm, wrist],
        "p3": [lead_upper, lead_forearm, shaft, wrist],
        "p4": [spine, shoulders, hips, shaft, lead_upper, lead_forearm, wrist, ("face", "face", "p4_face")],
        "p5": [hips, lead_thigh, trail_thigh, shaft, lead_upper, trail_upper],
        "p6": [shaft, hips, shoulders, wrist],
        "p7": [
            spine,
            shoulders,
            hips,
            lead_upper,
            trail_upper,
            lead_forearm,
            trail_forearm,
            lead_thigh,
            trail_thigh,
            lead_shin,
            trail_shin,
            shaft,
            wrist,
            ("face", "face", "p7_face"),
        ],
        "p8": [lead_upper, trail_upper, lead_forearm, trail_forearm, shaft, wrist_cross],
        "p9": [lead_upper, lead_forearm, shoulders, hips, lead_thigh, trail_thigh],
        "p10": [shoulders, hips, hands_by_head],
    }


def load_swing(folder: Path) -> dict | None:
    arrays = sorted(folder.glob("*_3d.npz"))
    summaries = sorted(folder.glob("*_3d.json"))
    if not arrays or not summaries:
        return None
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    data = np.load(arrays[0])
    return {"folder": folder, "summary": summary, "poses": data["keypoints_3d"]}


def collect_standards(folder: Path) -> list[dict]:
    swings = []
    direct = load_swing(folder)
    if direct is not None:
        swings.append(direct)
    for child in sorted(path for path in folder.iterdir() if path.is_dir()):
        swing = load_swing(child)
        if swing is not None:
            swings.append(swing)
    return swings


def frame_of(summary: dict, key: str) -> int | None:
    positions = summary.get("positions") or {}
    if key in positions and positions[key] is not None:
        return int(positions[key])
    phases = summary.get("phases") or {}
    fallback = {"p1": "address", "p4": "top", "p7": "impact"}.get(key)
    if fallback and phases.get(fallback) is not None:
        return int(phases[fallback])
    return None


def pose_at(swing: dict, key: str) -> np.ndarray | None:
    frame = frame_of(swing["summary"], key)
    poses = swing["poses"]
    if frame is None or not (0 <= frame < len(poses)):
        return None
    return poses[frame]


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second))


def direction_score(value: float) -> float:
    return round((value + 1.0) / 2.0 * 100.0, 1)


def angle_score(user_angle: float, reference_angle: float) -> float:
    gap = abs(user_angle - reference_angle)
    return round(100.0 - min(100.0, gap / 180.0 * 100.0), 1)


def joint_direction(pose: np.ndarray, start_name: str, end_name: str) -> np.ndarray | None:
    start = pose[JOINT[start_name]]
    end = pose[JOINT[end_name]]
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return None
    return unit(start, end)


def direction_of(swing: dict, pose: np.ndarray, spec: tuple) -> np.ndarray | None:
    if spec[0] == "limb":
        start_name, end_name = limbs(lead_of(swing["summary"]))[spec[2]]
    else:
        start_name, end_name = spec[2], spec[3]
    return joint_direction(pose, start_name, end_name)


def wrist_angle(swing: dict, key: str) -> float | None:
    return wrist_shaft_degrees(swing["poses"], frame_of(swing["summary"], key), lead_of(swing["summary"]))


def face_angle(swing: dict, field: str) -> float | None:
    if swing["summary"].get("view") != "down_the_line":
        return None
    record = swing["summary"].get(field)
    if not isinstance(record, dict) or record.get("from_shaft_deg") is None:
        return None
    return float(record["from_shaft_deg"])


def mean_direction(vectors: list[np.ndarray]) -> np.ndarray | None:
    mean = np.mean(vectors, axis=0)
    return unit(np.zeros(3), mean)


def resample(path: np.ndarray, count: int) -> np.ndarray | None:
    finite = path[np.isfinite(path).all(axis=1)]
    if len(finite) < 2:
        return None
    distances = np.linalg.norm(np.diff(finite, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] < 1e-6:
        return None
    targets = np.linspace(0.0, cumulative[-1], count)
    sampled = np.column_stack([np.interp(targets, cumulative, finite[:, axis]) for axis in range(3)])
    sampled -= sampled.mean(axis=0)
    scale = float(np.linalg.norm(sampled))
    if scale < 1e-6:
        return None
    return sampled / scale


def path_score(user: dict, standards: list[dict]) -> float | None:
    def hosel_path(swing: dict) -> np.ndarray | None:
        phases = swing["summary"].get("phases") or {}
        start = phases.get("address")
        stop = phases.get("follow_through")
        if start is None or stop is None:
            return None
        return resample(swing["poses"][int(start) : int(stop) + 1, JOINT["hosel"]], PATH_SAMPLES)

    user_path = hosel_path(user)
    standard_paths = [path for path in (hosel_path(swing) for swing in standards) if path is not None]
    if user_path is None or not standard_paths:
        return None
    mean = np.mean(standard_paths, axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-6)
    similarity = cosine(user_path.ravel(), mean.ravel())
    return direction_score(similarity)


def score_parameter(user: dict, standards: list[dict], key: str, spec: tuple) -> float | None:
    if spec[0] == "face":
        user_angle = face_angle(user, spec[2])
        reference_angles = [angle for angle in (face_angle(swing, spec[2]) for swing in standards) if angle is not None]
        if user_angle is None or not reference_angles:
            return None
        return angle_score(user_angle, float(np.mean(reference_angles)))

    if spec[0] == "wrist":
        user_angle = wrist_angle(user, key)
        reference_angles = [angle for angle in (wrist_angle(swing, key) for swing in standards) if angle is not None]
        if user_angle is None or not reference_angles:
            return None
        return angle_score(user_angle, float(np.mean(reference_angles)))

    user_pose = pose_at(user, key)
    if user_pose is None:
        return None
    user_direction = direction_of(user, user_pose, spec)
    reference_directions = []
    for swing in standards:
        pose = pose_at(swing, key)
        if pose is None:
            continue
        direction = direction_of(swing, pose, spec)
        if direction is not None:
            reference_directions.append(direction)
    if user_direction is None or not reference_directions:
        return None
    reference = mean_direction(reference_directions)
    if reference is None:
        return None
    return direction_score(cosine(user_direction, reference))


def score_user(user: dict, standards: list[dict]) -> dict:
    checklist = criteria()
    positions = {}
    parameter_scores = []
    for key in POSITIONS:
        details = {}
        values = []
        for spec in checklist[key]:
            score = score_parameter(user, standards, key, spec)
            if score is None:
                continue
            details[spec[1]] = score
            values.append(score)
            parameter_scores.append(score)
        positions[key] = None if not values else {"score": round(float(np.mean(values)), 1), "parameters": details}
    total = None if not parameter_scores else round(float(np.mean(parameter_scores)), 1)
    return {
        "pose_score": total,
        "weight": "equal",
        "positions": positions,
        "path_score": path_score(user, standards),
        "standards": [swing["folder"].name for swing in standards],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a lifted swing against lifted standard swings.")
    parser.add_argument("--standards", type=Path, required=True, help="Folder of golf_pose_3d outputs.")
    parser.add_argument("--user", type=Path, required=True, help="One golf_pose_3d output folder.")
    args = parser.parse_args()
    if not args.standards.is_dir():
        raise SystemExit(f"Standards folder not found: {args.standards}")
    if not args.user.is_dir():
        raise SystemExit(f"User swing folder not found: {args.user}")
    standards = [swing for swing in collect_standards(args.standards) if swing["folder"].resolve() != args.user.resolve()]
    user = load_swing(args.user)
    if user is None:
        raise SystemExit(f"No *_3d.npz in {args.user}")
    if not standards:
        raise SystemExit(f"No standard swings found in {args.standards}")
    result = score_user(user, standards)
    destination = args.user / "score.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Pose score: {result['pose_score']}")
    for key, position in result["positions"].items():
        score = None if position is None else position["score"]
        print(f"  {key}: {score}")
    print(f"Path score: {result['path_score']}")
    print(f"Score: {destination}")


if __name__ == "__main__":
    main()
