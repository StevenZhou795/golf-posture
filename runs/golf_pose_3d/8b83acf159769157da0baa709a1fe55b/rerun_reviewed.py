"""Per-video reviewed checkpoints; leaves the shared workflow code unchanged.

The automatic club-based phase detector failed on this low-light clip.
Frame numbers are zero-based. P7 is the nearest visible impact frame at 30 fps,
not a measured subframe contact time. Run from any working directory.
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

FOLDER = Path(__file__).resolve().parent
STEM = FOLDER.name
VIDEO = ROOT / "pro_refs" / (STEM + ".mp4")
POSITIONS = dict(zip((f"p{i}" for i in range(1, 11)), [39, 46, 51, 59, 63, 65, 66, 68, 70, 80]))


def main():
    summary_path = FOLDER / f"{STEM}_3d.json"
    automatic = FOLDER / "automatic_detection.json"
    if summary_path.exists() and not automatic.exists():
        shutil.copy2(summary_path, automatic)
    # Overrides apply only in this process and only to this reviewed video.
    workflow.detect_phases = lambda *_args, **_kwargs: SwingPhases(39, 59, 66, 80, "visual_review_low_light")
    workflow.impact_from_ball = lambda *_args, **_kwargs: POSITIONS["p7"]
    workflow.find_checkpoint_frames = lambda *_args, **_kwargs: POSITIONS.copy()
    sys.argv = ["golf_pose_3d.py", str(VIDEO), "--view", "down_the_line"]
    workflow.main()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_selection"] = {
        "method": "visual_review_after_automatic_phase_failure",
        "frame_index_base": 0,
        "reason": "Low-light club segmentation gaps caused inconsistent automatic phases and missing positions.",
        "impact_precision": "Nearest source frame, approximately +/-1 frame at 30 fps; contact time not independently measured.",
        "shared_workflow_code_modified": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raw = FOLDER / "raw_frames"
    raw.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    tiles = []
    records = []
    try:
        for name, index in POSITIONS.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read {name} frame {index}")
            cv2.imwrite(str(raw / f"{name.upper()}.jpg"), frame)
            tile = cv2.copyMakeBorder(frame, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, f"{name.upper()}  f{index}  {index/30:.3f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (20, 20, 20), 1, cv2.LINE_AA)
            tiles.append(tile)
            records.append({"position": name.upper(), "frame_index": index, "time_s": round(index/30, 4),
                            "raw_frame": f"raw_frames/{name.upper()}.jpg", "annotated_frame": f"{name.upper()}.jpg"})
    finally:
        capture.release()
    cv2.imwrite(str(FOLDER / "P1_P10_overview.jpg"), np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])]))
    (FOLDER / "selected_frames.json").write_text(json.dumps({
        "source": str(VIDEO), "fps": 30, "frame_index_base": 0,
        "selection": "visually_reviewed", "frames": records}, indent=2), encoding="utf-8")
    print("Verified export of ten reviewed checkpoints and original frames.")


if __name__ == "__main__":
    main()
