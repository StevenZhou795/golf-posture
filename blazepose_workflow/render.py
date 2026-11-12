"""Skeleton overlay video, 2D position stills and 3D skeleton stills."""
from pathlib import Path

import cv2
import numpy as np

from blazepose_workflow.landmarks import EDGE_INDEX, J, NAMES, side

POSITION_NAMES = {
    "p1": "address", "p2": "takeaway", "p3": "halfway back", "p4": "top", "p5": "early downswing",
    "p6": "pre-impact", "p7": "impact", "p8": "release", "p9": "finish (arm parallel)", "p10": "end",
}
COLOURS = {"left": (80, 220, 80), "right": (40, 140, 255), "center": (255, 180, 60)}


def _side_of_edge(a: int, b: int) -> str:
    sides = {side(NAMES[a]), side(NAMES[b])}
    return sides.pop() if len(sides) == 1 else "center"


def draw_pose(image: np.ndarray, pose: np.ndarray) -> None:
    """Draw one (23, 2) pose in pixels; NaN landmarks are skipped."""
    thickness = max(2, image.shape[0] // 400)
    for a, b in EDGE_INDEX:
        if np.isfinite(pose[[a, b]]).all():
            cv2.line(image, tuple(np.round(pose[a]).astype(int)), tuple(np.round(pose[b]).astype(int)),
                     COLOURS[_side_of_edge(a, b)], thickness, cv2.LINE_AA)
    for index, point in enumerate(pose):
        if np.isfinite(point).all():
            radius = thickness + (2 if index == J["nose"] else 1)
            cv2.circle(image, tuple(np.round(point).astype(int)), radius, COLOURS[side(NAMES[index])], -1,
                       cv2.LINE_AA)


def label(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.8) -> None:
    # A box rather than a thick outline: Hershey glyph advance grows with thickness,
    # so an outline pass drifts away from the text pass.
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    cv2.rectangle(image, (x - 6, y - height - 6), (x + width + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def render_outputs(video: Path, folder: Path, poses2d: np.ndarray, world: np.ndarray,
                   positions: dict[str, int] | None, fps: float) -> list[str]:
    """Write `<stem>_blazepose.mp4`, P*.jpg and P*_3d.jpg; returns the written file names."""
    by_frame = {frame: key for key, frame in (positions or {}).items()}
    capture = cv2.VideoCapture(str(video))
    writer = None
    written = []
    current = None
    try:
        for index in range(len(poses2d)):
            ok, frame = capture.read()
            if not ok:
                break
            draw_pose(frame, poses2d[index])
            if index in by_frame:
                current = by_frame[index]
                still = frame.copy()
                label(still, f"{current.upper()} {POSITION_NAMES[current]}  frame {index}",
                      (16, still.shape[0] - 24))
                path = folder / f"{current.upper()}.jpg"
                cv2.imwrite(str(path), still)
                written.append(path.name)
            label(frame, f"frame {index}" + (f"  {current.upper()} {POSITION_NAMES[current]}" if current else ""),
                  (16, 36), 0.7)
            if writer is None:
                path = folder / f"{video.stem}_blazepose.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0,
                                         (frame.shape[1], frame.shape[0]))
                written.append(path.name)
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    for key, frame in (positions or {}).items():
        path = folder / f"{key.upper()}_3d.jpg"
        save_3d(world[frame], path, f"{key.upper()} {POSITION_NAMES[key]}  frame {frame}")
        written.append(path.name)
    return written


def save_3d(pose: np.ndarray, path: Path, title: str) -> None:
    """Three views of one (23, 3) world pose (metres, hip origin): front, side and top-down 3D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Camera axes (x right, y down, z away) to plot axes (x right, y depth, z up).
    plot = np.stack([pose[:, 0], pose[:, 2], -pose[:, 1]], axis=1)
    finite = np.isfinite(plot).all(axis=1)
    if finite.any():
        low, high = np.nanmin(plot[finite], axis=0), np.nanmax(plot[finite], axis=0)
        centre, half = (low + high) / 2, max(0.3, float((high - low).max()) / 2 * 1.1)
    else:
        centre, half = np.zeros(3), 1.0
    figure = plt.figure(figsize=(9, 4.2), dpi=110)
    for slot, (elevation, azimuth, name) in enumerate([(5, -90, "camera view"), (5, 0, "side view"),
                                                       (25, -60, "3D")], 1):
        axes = figure.add_subplot(1, 3, slot, projection="3d")
        for a, b in EDGE_INDEX:
            if np.isfinite(plot[[a, b]]).all():
                colour = np.array(COLOURS[_side_of_edge(a, b)][::-1]) / 255
                axes.plot(*plot[[a, b]].T, color=colour, linewidth=2)
        axes.scatter(*plot[finite].T, s=6, color="k")
        axes.set_xlim(centre[0] - half, centre[0] + half)
        axes.set_ylim(centre[1] - half, centre[1] + half)
        axes.set_zlim(centre[2] - half, centre[2] + half)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_title(name, fontsize=9)
        axes.set_xticklabels([]); axes.set_yticklabels([]); axes.set_zticklabels([])
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
