"""python -m blazepose_workflow video <path> [options]"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blazepose_workflow",
                                     description="Golf pose inference with MediaPipe BlazePose GHUM (body only).")
    commands = parser.add_subparsers(dest="command", required=True)
    video = commands.add_parser("video", help="2D + 3D pose and P1-P10 for one swing video")
    video.add_argument("path", type=Path)
    video.add_argument("--model", choices=("lite", "full", "heavy"), default="heavy")
    video.add_argument("--view", choices=("auto", "face_on", "down_the_line"), default="auto")
    video.add_argument("--handedness", choices=("auto", "right", "left"), default="auto")
    video.add_argument("--golfer-box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                       help="pixel box containing the golfer's hips on the first frames")
    video.add_argument("--no-crop", action="store_true", help="skip the second pass on the golfer ROI")
    video.add_argument("--out", type=Path, help="output folder (default runs/blazepose_workflow/videos/<stem>_<time>)")
    args = parser.parse_args(argv)
    from blazepose_workflow.pipeline import process
    try:
        process(args.path, args.model, args.view, args.handedness, args.golfer_box, not args.no_crop, args.out)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
