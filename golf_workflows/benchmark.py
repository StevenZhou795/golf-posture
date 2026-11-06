"""P1-P10 frame benchmark: replay saved 2D through `process`, then score positions against labels.

Labels (benchmarks/p_position_labels.json) give an inclusive frame range per
position. Error is zero inside the range, otherwise the distance in frames to
the nearest labelled frame; milliseconds use the clip frame rate.

    python -m golf_workflows.benchmark evaluate RUN_DIR [RUN_DIR ...]
    python -m golf_workflows.benchmark replay --out runs/benchmark/NAME [--clips w09 005]

Replay patches 2D detection with the arrays saved in each clip's
`replay_run`, so selection, tracking and lifting changes can be compared
without re-running the 2D models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "benchmarks" / "p_position_labels.json"
POSITIONS = [f"p{i}" for i in range(1, 11)]


def load_labels(path: Path = LABELS) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("frame_index_base", 0) != 0:
        raise ValueError("Labels must be zero-based")
    return data["clips"]


def position_errors(positions: dict, labels: dict) -> dict[str, int | None]:
    """Frames outside the labelled range per labelled position (None if not predicted)."""
    errors = {}
    for key, (first, last) in labels.items():
        frame = positions.get(key)
        if frame is None:
            errors[key] = None
        else:
            errors[key] = 0 if first <= frame <= last else int(min(abs(frame - first), abs(frame - last)))
    return errors


def evaluate_summary(summary: dict, clip_labels: dict) -> dict:
    fps = float(summary.get("fps") or 30.0)
    errors = position_errors(summary["positions"], clip_labels.get("labels", {}))
    selection = summary.get("checkpoint_selection", {})
    return {
        "positions": summary["positions"],
        "errors_frames": errors,
        "errors_ms": {key: None if value is None else round(1000.0 * value / fps, 1) for key, value in errors.items()},
        "confidence": selection.get("position_confidence"),
        "subframe": selection.get("subframe_frames"),
        "view": summary.get("view"), "view_expected": clip_labels.get("view"),
        "handedness": summary.get("handedness"), "handedness_expected": clip_labels.get("handedness"),
        "fps": fps,
    }


def aggregate(results: dict[str, dict]) -> dict:
    rows = [(clip, key, value) for clip, result in results.items() for key, value in result["errors_frames"].items()]
    scored = [value for _, _, value in rows if value is not None]
    by_position = {}
    for key in POSITIONS:
        values = [value for _, k, value in rows if k == key and value is not None]
        if values:
            by_position[key] = {"n": len(values), "mean_frames": round(float(np.mean(values)), 2),
                                "within_range": round(float(np.mean([v == 0 for v in values])), 2)}
    return {
        "labelled": len(rows),
        "missing_predictions": sum(value is None for _, _, value in rows),
        "mean_error_frames": round(float(np.mean(scored)), 3) if scored else None,
        "within_range": round(float(np.mean([v == 0 for v in scored])), 3) if scored else None,
        "within_1": round(float(np.mean([v <= 1 for v in scored])), 3) if scored else None,
        "within_2": round(float(np.mean([v <= 2 for v in scored])), 3) if scored else None,
        "view_correct": sum(r["view"] == r["view_expected"] for r in results.values() if r["view_expected"]),
        "handedness_correct": sum(r["handedness"] == r["handedness_expected"]
                                  for r in results.values() if r["handedness_expected"]),
        "clips": len(results),
        "by_position": by_position,
    }


def _summary_in(folder: Path) -> tuple[str, dict]:
    pairs = sorted(Path(folder).glob("*_3d.json"))
    if len(pairs) != 1:
        raise ValueError(f"Expected one *_3d.json in {folder}")
    return pairs[0].name.removesuffix("_3d.json"), json.loads(pairs[0].read_text(encoding="utf-8"))


def evaluate(folders: list[Path], labels_path: Path = LABELS) -> dict:
    labels = load_labels(labels_path)
    results = {}
    for folder in folders:
        clip, summary = _summary_in(folder)
        if clip not in labels:
            print(f"{clip}: no labels, skipped")
            continue
        results[clip] = evaluate_summary(summary, labels[clip])
    return {"clips": results, "aggregate": aggregate(results)}


def saved_swing(run: Path, stem: str):
    """Swing2D rebuilt from a completed run's arrays."""
    from golf_workflows import golf_pose_3d as g

    data = np.load(run / f"{stem}_3d.npz")
    points, valid = data["keypoints_2d"].astype(np.float64), data["keypoints_2d_valid"]
    boxes = []
    for p, v in zip(points, valid):
        body = p[:17][v[:17]]
        boxes.append(np.concatenate([body.min(0), body.max(0)]) if len(body) >= 2 else np.full(4, np.nan))
    wrist_y = np.array([np.nanmean([p[i, 1] for i in (13, 16) if v[i]]) if v[[13, 16]].any() else np.nan
                        for p, v in zip(points, valid)])
    return g.Swing2D(frames=len(points), fps=float(data["fps"]), size=tuple(int(x) for x in data["frame_size"]),
                     points=points, valid=valid, filled=g.fill_gaps(points, valid), boxes=np.stack(boxes),
                     samples=[], wrist_y=wrist_y, club_confidence=data["club_landmark_confidence"],
                     keypoint_confidence=data["keypoints_2d_confidence"] if "keypoints_2d_confidence" in data else None)


def replay(clips: list[str] | None, out: Path, labels_path: Path = LABELS, view: str = "auto",
           handedness: str = "auto", mask: bool = True) -> dict:
    from ultralytics import YOLO

    from golf_workflows import golf_pose_3d as g

    labels = load_labels(labels_path)
    lifter = g.load_lifter(g.CHECKPOINT)
    mask_model = YOLO(str(g.MODEL_PATH)) if mask and g.MODEL_PATH.is_file() else None
    original = g.detect_2d
    folders = []
    try:
        for clip in clips or list(labels):
            entry = labels[clip]
            run, video = ROOT / entry["replay_run"], ROOT / entry["video"]
            stem = video.stem
            summary = json.loads((run / f"{stem}_3d.json").read_text(encoding="utf-8"))
            swing = saved_swing(run, stem)
            g.detect_2d = lambda *args, _swing=swing, **kwargs: _swing
            pose_model = SimpleNamespace(joint_pose=True, metadata=summary.get("pose_backend") or {})
            folder = Path(out) / clip
            folder.mkdir(parents=True, exist_ok=True)
            g.process(video, pose_model, None, lifter, handedness=handedness, view=view, club_refinement=True,
                      output_folder=folder, club_mask_model=mask_model)
            folders.append(folder)
    finally:
        g.detect_2d = original
    return evaluate(folders, labels_path)


def report(result: dict) -> None:
    for clip, row in result["clips"].items():
        errors = ", ".join(f"{k}={v}" for k, v in row["errors_frames"].items()) or "no labels"
        print(f"{clip}: positions {[row['positions'][k] for k in POSITIONS]}; errors {errors}; "
              f"view {row['view']} (expected {row['view_expected']}); handedness {row['handedness']}")
    print("Aggregate:", json.dumps(result["aggregate"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="Score completed run folders")
    ev.add_argument("runs", type=Path, nargs="+")
    ev.add_argument("--json", type=Path, help="Write the report here")
    rp = sub.add_parser("replay", help="Replay saved 2D for labelled clips, then score")
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--clips", nargs="*")
    rp.add_argument("--view", default="auto", choices=("auto", "down_the_line", "face_on"))
    rp.add_argument("--handedness", default="auto", choices=("auto", "right", "left"))
    rp.add_argument("--no-club-mask", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate(args.runs, args.labels)
        target = args.json
    else:
        result = replay(args.clips, args.out, args.labels, args.view, args.handedness, not args.no_club_mask)
        target = args.out / "benchmark.json"
    report(result)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
