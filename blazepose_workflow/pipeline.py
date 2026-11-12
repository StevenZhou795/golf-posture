"""One video: BlazePose inference, cleaning, P1-P10, and every output file."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from blazepose_workflow import inference, smoothing
from blazepose_workflow.landmarks import NAMES
from blazepose_workflow.positions import POSITIONS, SwingCoverageError, select_positions
from blazepose_workflow.render import render_outputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "blazepose_workflow" / "videos"
# BlazePose predicts occluded joints too (low visibility, high presence). From a
# rear camera the lead wrist is hidden behind the body for most of the
# follow-through (005: frames 233-260), so only joints outside the image
# (low presence) are dropped; visibility is kept in `image_raw`.
PRESENT = 0.5


def clean_2d(track: inference.PoseTrack) -> np.ndarray:
    points = smoothing.gate(track.image[..., :2], track.image[..., 4], PRESENT)
    return smoothing.smooth(smoothing.fill_gaps(points, track.times_ms), track.times_ms)


def clean_3d(track: inference.PoseTrack) -> np.ndarray:
    points = smoothing.gate(track.world[..., :3], track.world[..., 4], PRESENT)
    points = smoothing.normalize_bones(smoothing.fill_gaps(points, track.times_ms))
    return smoothing.smooth(points, track.times_ms)


def process(video: Path, model: str = "heavy", view: str = "auto", handedness: str = "auto",
            golfer_box: list[float] | None = None, crop: bool = True, out: Path | None = None) -> Path:
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    folder = Path(out) if out else OUTPUT_DIR / f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Source: {video}\nOutput: {folder}", flush=True)
    track, detection = inference.detect(video, model, golfer_box, crop)
    print(f"Golfer found in {int(track.found.sum())}/{track.frames} frames "
          f"(ROI {detection.get('roi_xyxy')}, {detection.get('roi_pass')})", flush=True)
    points2d = clean_2d(track)
    world = clean_3d(track)
    selection, error = None, None
    try:
        selection = select_positions(points2d, world, track.times_ms, view, handedness)
    except SwingCoverageError as caught:
        error = str(caught)
        print(f"ERROR: {error}", flush=True)
    positions = None if selection is None else selection["positions"]
    np.savez_compressed(
        folder / f"{video.stem}_pose.npz", joints=np.array(NAMES), times_ms=track.times_ms,
        image_raw=track.image, world_raw=track.world, image_2d=points2d, world_3d=world,
        found=track.found, candidates=track.candidates)
    written = render_outputs(video, folder, points2d, world, positions, track.fps)
    gaps = ~np.isfinite(points2d).all(axis=2)
    occluded = track.found[:, None] & ~(track.image[..., 3] >= smoothing.VISIBLE)
    summary = {
        "video": str(video), "frames": track.frames, "nominal_fps": round(track.fps, 3),
        "size": list(track.size), "detection": detection,
        "frame_source": {s: track.source.count(s) for s in set(track.source)},
        "landmarks": NAMES, "dropped_landmarks": "BlazePose 1-10 (face) and all club inference",
        "coordinates": {"image_2d": "pixels, y down",
                        "world_3d": "metres, hip centre at origin, camera axes: x right, y down, "
                                    "z away from the camera (BlazePose GHUM world landmarks)"},
        "cleaning": {"presence_2d_and_3d": PRESENT, "occluded_below_visibility": smoothing.VISIBLE,
                     "max_gap_s": smoothing.MAX_GAP_SECONDS, "smooth_s": smoothing.SMOOTH_SECONDS,
                     "bone_lengths_normalised": True},
        "missing_2d_landmarks_per_frame": gaps.sum(axis=1).tolist(),
        "occluded_2d_landmarks_per_frame": occluded.sum(axis=1).tolist(),
        "error": error, **(selection or {}), "files": written,
    }
    (folder / f"{video.stem}_pose.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if positions is not None:
        (folder / "identified_frames.json").write_text(json.dumps({
            "video": str(video), "positions": positions,
            "times_ms": {key: selection["details"][key]["time_ms"] for key in POSITIONS},
        }, indent=2), encoding="utf-8")
        print(f"Positions: {positions}", flush=True)
        print(f"View: {selection['view']}, handedness: {selection['handedness']}", flush=True)
    print(f"Wrote {len(written) + 2 + (positions is not None)} files to {folder}", flush=True)
    if error:
        raise SwingCoverageError(error)
    return folder
