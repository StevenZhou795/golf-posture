"""A face-on synthetic swing driven by one lead-arm angle profile."""
import numpy as np

from blazepose_workflow.landmarks import J, NAMES

# Frame -> lead-arm angle from straight down: address hold, top at 40, impact at 55, finish hold from 75.
PROFILE = ([0, 15, 40, 55, 75, 99], [10, 10, 120, 10, 150, 150])


def swing(handedness: str = "right", n: int = 100, fps: float = 30.0):
    """(points2d px, world metres, times_ms) for a golfer facing the camera."""
    angle = np.radians(np.interp(np.arange(n), *PROFILE))
    world = np.zeros((n, len(NAMES), 3))
    fixed = {"left_hip": (.1, 0), "right_hip": (-.1, 0), "left_shoulder": (.18, -.5),
             "right_shoulder": (-.18, -.5), "left_knee": (.12, .45), "right_knee": (-.12, .45),
             "left_ankle": (.14, .9), "right_ankle": (-.14, .9), "left_heel": (.14, .93),
             "right_heel": (-.14, .93), "left_foot_index": (.2, .95), "right_foot_index": (-.2, .95),
             "nose": (0, -.72)}
    for name, (x, y) in fixed.items():
        world[:, J[name], :2] = (x, y)
    # Backswing toward the trail side (image left for a right-hander), follow-through to the lead side.
    direction = np.where(np.arange(n) <= 47, -1.0, 1.0)
    for shoulder in ("left_shoulder", "right_shoulder"):
        base = world[:, J[shoulder], :2]
        hand = np.stack([base[:, 0] * 0 + direction * .6 * np.sin(angle), base[:, 1] + .6 * np.cos(angle)], axis=1)
        side = shoulder.split("_")[0]
        world[:, J[f"{side}_wrist"], :2] = hand
        world[:, J[f"{side}_elbow"], :2] = (base + hand) / 2
        for part in ("pinky", "index", "thumb"):
            world[:, J[f"{side}_{part}"], :2] = hand + [0, .05]
    if handedness == "left":
        world[..., 0] *= -1
        swap = [J[name] if not name.startswith(("left_", "right_")) else
                J[("right_" + name[5:]) if name.startswith("left_") else ("left_" + name[6:])] for name in NAMES]
        world = world[:, swap]
    points2d = 500 + 200 * world[..., :2]
    return points2d, world, np.arange(n) * 1000 / fps
