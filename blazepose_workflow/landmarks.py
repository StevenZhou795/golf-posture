"""Body-only subset of the 33 BlazePose landmarks, skeleton edges and derived points.

Face landmarks 1-10 (eyes, ears, mouth) are dropped. The nose is kept as a
head reference for drawing and for the facing direction.
"""
import warnings

import numpy as np

# BlazePose index -> name, for the kept landmarks, in output order.
BLAZEPOSE = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}
SOURCE_INDEX = list(BLAZEPOSE)
NAMES = list(BLAZEPOSE.values())
J = {name: index for index, name in enumerate(NAMES)}

EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_wrist", "left_index"), ("left_wrist", "left_pinky"), ("left_wrist", "left_thumb"),
    ("right_wrist", "right_index"), ("right_wrist", "right_pinky"), ("right_wrist", "right_thumb"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
]
EDGE_INDEX = [(J[a], J[b]) for a, b in EDGES]
# Bones whose length is held to the clip median in 3D (hands and feet are too short to matter).
BONES = [edge for edge in EDGE_INDEX
         if not any(part in NAMES[i] for i in edge for part in ("pinky", "index", "thumb", "heel", "foot"))]


def side(name: str) -> str:
    return "left" if name.startswith("left_") else "right" if name.startswith("right_") else "center"


def lead_trail(handedness: str) -> tuple[str, str]:
    """(lead, trail) body side; a right-handed golfer leads with the left side."""
    if handedness not in ("right", "left"):
        raise ValueError(f"handedness must be 'right' or 'left', not {handedness!r}")
    return ("left", "right") if handedness == "right" else ("right", "left")


def subset(landmarks) -> np.ndarray:
    """(23, 5) x, y, z, visibility, presence for the kept landmarks of one MediaPipe pose."""
    return np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z,
                      landmarks[i].visibility or 0.0, landmarks[i].presence or 0.0] for i in SOURCE_INDEX])


def midpoint(points: np.ndarray, a: str, b: str) -> np.ndarray:
    return (points[..., J[a], :] + points[..., J[b], :]) / 2


def hip_center(points: np.ndarray) -> np.ndarray:
    return midpoint(points, "left_hip", "right_hip")


def shoulder_center(points: np.ndarray) -> np.ndarray:
    return midpoint(points, "left_shoulder", "right_shoulder")


def hands(points: np.ndarray) -> np.ndarray:
    """Grip position: mean of both wrists, index and thumb points (NaN-aware)."""
    names = [f"{s}_{part}" for s in ("left", "right") for part in ("wrist", "index", "thumb")]
    stack = np.stack([points[..., J[name], :] for name in names])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # frames with no hand point stay NaN
        return np.nanmean(stack, axis=0)


def torso_length(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(shoulder_center(points) - hip_center(points), axis=-1)
