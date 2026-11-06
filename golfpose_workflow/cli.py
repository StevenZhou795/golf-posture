"""Fresh joint 22-point 2D inference, 17+2 lifting, with its own references, videos, scores and pointers."""
from __future__ import annotations

import argparse
from pathlib import Path

from golf_workflows import cli as shared

RUNS = shared.ROOT / 'runs' / 'golfpose_workflow'
BACKEND = 'golfpose-golfer-vitpose-huge'
# heel/toe_down/toe_up are dropped: sub-heatmap-cell on the clubhead and unused by phases or scoring.
LIFTER = '17+2'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reference', action='store_true', help='Infer all ten reference sheets from scratch')
    mode.add_argument('--video', type=Path, metavar='VIDEO', help='Infer a video and extract P1-P10')
    mode.add_argument('--score', action='store_true', help='Score the saved ten poses against ten references')
    parser.add_argument('--input-dir', type=Path, default=shared.ROOT / 'pro_refs' / 'Standard_10_frames',
                        help='P1-P10 reference sheet folder')
    parser.add_argument('--run', type=Path, help='Video workflow folder for scoring; defaults to this workflow latest')
    parser.add_argument('--references', type=Path, help='Reference folder for scoring; defaults to this workflow latest')
    parser.add_argument('--output-dir', type=Path, help='Optional empty output folder')
    parser.add_argument('--pose-cpu-threads', type=int, default=8)
    parser.add_argument('--positions-json', type=Path, help='Optional reviewed P1-P10 indices; otherwise automatic')
    parser.add_argument('--golfer-box', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='Golfer-with-club ROI in original-frame pixels where YOLOX misses; '
                             'overrides the automatic YOLO26n ROI')
    parser.add_argument('--no-auto-golfer-box', action='store_true',
                        help='Do not derive a default golfer ROI from a YOLO26n pass over sampled frames')
    parser.add_argument('--handedness', choices=('auto', 'right', 'left'), default='auto',
                        help='auto: from the 2D hands; falls back to right when ambiguous')
    parser.add_argument('--view', choices=('auto', 'down_the_line', 'face_on'), default='auto',
                        help='auto: from shoulder width and the feet line; falls back to down_the_line')
    parser.add_argument('--no-body-fusion', action='store_true',
                        help='Do not fill low-confidence ViTPose body joints from YOLO26n')
    parser.add_argument('--no-club-mask', action='store_true',
                        help='Do not use the golf-club segmentation mask as a shaft-angle measurement')
    parser.add_argument('--allow-skeleton-gaps', action='store_true',
                        help='Warn instead of aborting when the golfer skeleton is lost inside the swing window')
    parser.add_argument('--no-club-refinement', action='store_true')
    parser.add_argument('--club-marker-length-m', type=float)
    parser.add_argument('--track-club-path', action='store_true')
    args = parser.parse_args(argv)
    if args.pose_cpu_threads < 1:
        parser.error('--pose-cpu-threads must be positive')
    if not args.score and (args.run or args.references):
        parser.error('--run and --references apply only to --score')
    if args.video is None and (args.positions_json or args.golfer_box is not None or args.no_auto_golfer_box
                              or args.no_club_refinement or args.no_body_fusion or args.no_club_mask
                              or args.allow_skeleton_gaps
                              or args.club_marker_length_m is not None or args.track_club_path):
        parser.error('Frame selection and club refinement options require --video')
    # Fresh inference only: no --from-run and no dependency on the YOLO workflow pointers.
    args.runs_root = RUNS
    args.from_run = None
    args.pose_backend = BACKEND
    args.lifter = LIFTER
    # Shared entry points ignore these legacy flags for the joint backend.
    args.club_pose_backend = 'vitpose-huge'
    args.club_crop_backend = 'golfer-box' if args.reference else 'segmentation'
    try:
        if args.reference:
            shared.references(args)
        elif args.video is not None:
            shared.video(args)
        else:
            shared.score(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
