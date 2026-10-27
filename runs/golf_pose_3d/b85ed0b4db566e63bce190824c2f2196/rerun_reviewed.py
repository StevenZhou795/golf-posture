"""Re-export this video with visually reviewed positions and cached detections.

Shared workflow source is unchanged. Frame indices are zero-based; source
timestamps describe the edited slow-motion video, not real-time swing speed.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import golf_pose_3d as workflow
from swing_phases import SwingPhases
from trace_swing_plane import ClubSample

FOLDER = Path(__file__).resolve().parent
STEM = FOLDER.name
VIDEO = ROOT / "pro_refs" / f"{STEM}.mp4"
POSITIONS = dict(zip((f"p{i}" for i in range(1, 11)), [60, 88, 114, 172, 212, 226, 237, 254, 280, 340]))


def main():
    summary_path, arrays_path = FOLDER/f"{STEM}_3d.json", FOLDER/f"{STEM}_3d.npz"
    for source, backup in ((summary_path, FOLDER/"automatic_detection.json"), (arrays_path, FOLDER/"automatic_detection.npz")):
        if not backup.exists():
            shutil.copy2(source, backup)
    original = json.loads((FOLDER/"automatic_detection.json").read_text(encoding="utf-8"))
    with np.load(FOLDER/"automatic_detection.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    points, valid = data["keypoints_2d"], data["keypoints_2d_valid"]
    samples, previous = [], None
    for frame in np.flatnonzero(valid[:, workflow.JOINT["hosel"]]):
        grip = points[frame, workflow.JOINT["shaft"]] if valid[frame, workflow.JOINT["shaft"]] else None
        samples.append(ClubSample(int(frame), grip, points[frame, workflow.JOINT["hosel"]],
                                  0.0, previous is not None and frame-previous <= 6))
        previous = int(frame)
    swing = workflow.Swing2D(len(points), float(data["fps"]), tuple(data["frame_size"].astype(int)),
                            points, valid, data["keypoints_2d_filled"], np.zeros((len(points), 4)),
                            samples, np.zeros(len(points)))
    # Use original raw 2D and normalization; do not re-smooth cached filled data.
    workflow.detect_2d = lambda *_args, **_kwargs: swing
    workflow.smooth_keypoints = lambda points, _fps: points
    workflow.golf_canvas = lambda _swing: (data["keypoints_2d_normalized"], original["normalization"])
    workflow.detect_phases = lambda *_args, **_kwargs: SwingPhases(60, 172, 237, 340, "visual_review")
    workflow.track_ball = lambda *_args, **_kwargs: original["ball"].copy()
    workflow.impact_from_ball = lambda *_args, **_kwargs: POSITIONS["p7"]
    workflow.find_checkpoint_frames = lambda *_args, **_kwargs: POSITIONS.copy()
    workflow.process(VIDEO, None, None, workflow.load_lifter(workflow.CHECKPOINT), view="down_the_line")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"] = {
        "method": "visual_review_after_automatic_phase_failure", "frame_index_base": 0,
        "reason": "Automatic phases misplaced the top and returned no impact, leaving only P1/P4.",
        "impact_precision": "Nearest visible source frame, approximately +/-1 frame; not measured contact time.",
        "timebase": "Edited slow-motion video timeline, not physical swing time",
        "cached_detections_reused": True, "shared_workflow_code_modified": False,
        "visualization_note": "Club path reconstructed from saved detected points; unavailable original sample confidences are not inferred."
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw = FOLDER/"raw_frames"
    raw.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    tiles, records = [], []
    try:
        for name, index in POSITIONS.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {name} at frame {index}")
            cv2.imwrite(str(raw/f"{name.upper()}.jpg"), image)
            tile = cv2.copyMakeBorder(image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
            cv2.putText(tile, f"{name.upper()} f{index} {index/float(data['fps']):.3f}s", (8,22), cv2.FONT_HERSHEY_SIMPLEX, .55, (20,20,20), 1, cv2.LINE_AA)
            tiles.append(tile)
            records.append({"position": name.upper(), "frame_index": index,
                            "time_s": round(index/float(data["fps"]), 4),
                            "raw_frame": f"raw_frames/{name.upper()}.jpg", "annotated_frame": f"{name.upper()}.jpg"})
    finally:
        capture.release()
    cv2.imwrite(str(FOLDER/"P1_P10_overview.jpg"), np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
    (FOLDER/"selected_frames.json").write_text(json.dumps({
        "source": str(VIDEO), "fps": float(data["fps"]), "frame_index_base": 0,
        "selection": "visually_reviewed", "timebase": "edited_slow_motion_video", "frames": records}, indent=2), encoding="utf-8")
    print("Exported ten reviewed checkpoints, raw frames and overview.")


if __name__ == "__main__":
    main()
