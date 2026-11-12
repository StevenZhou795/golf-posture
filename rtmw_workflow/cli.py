import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time
import traceback

from .runtime import HERE, ROOT, CHECKPOINT
from .report import write_json


def inputs(args):
    if args.command == 'video':
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        return [args.video.resolve()]
    videos = sorted(p.resolve() for p in args.input_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'})
    if not videos:
        raise ValueError(f'No videos in {args.input_dir}')
    if len({p.stem.lower() for p in videos}) != len(videos):
        raise ValueError('Input video stems must be unique')
    priority = getattr(args, 'priority', None) or []
    missing = set(priority) - {p.name for p in videos}
    if missing:
        raise ValueError(f'Priority videos not found: {sorted(missing)}')
    videos.sort(key=lambda p: (priority.index(p.name) if p.name in priority else len(priority), p.name))
    return videos


def load_components(args):
    from ultralytics import YOLO, YOLOE
    from golf_workflows import golf_pose_3d as shared
    from golf_workflows.official_yolox import OfficialClubDetector
    from .runtime import load_model
    from .backend import RTMWBodyPose
    from .swingtrace import SwingTraceClub
    detector = OfficialClubDetector()
    model = load_model(args.checkpoint, args.cpu_threads)
    body = RTMWBodyPose(model, detector, args.checkpoint.resolve(), threads=args.cpu_threads)
    ball = YOLOE(str(shared.BALL_MODEL))
    ball.overrides['device'] = 'cpu'
    ball.set_classes(['golf ball'])
    roi = YOLO(str(ROOT / 'yolo26n-pose.pt'))
    roi.overrides['device'] = 'cpu'
    return dict(body=body, detector=detector, model=model,
        lifter=shared.load_lifter(shared.CHECKPOINT),
        club=SwingTraceClub(args.club_checkpoint,
            'cpu' if getattr(args, 'pose_device', 'cpu') == 'cuda' else args.club_device), ball=ball,
        roi=roi)


def run_video(video, folder, components, args):
    from golf_workflows.golf_pose_3d import process
    from golf_workflows.golfer_roi import auto_golfer_roi
    from .report import export, render_wholebody
    from .swingtrace import SwingTraceSequence
    body = components['body']
    body.records = []
    body.replay_results = None
    components['club'].reset()
    if args.golfer_box is not None:
        body.set_initial_golfer_box(args.golfer_box)
    else:
        roi, details = auto_golfer_roi(video, components['roi'])
        body.set_initial_golfer_box(roi, 'auto-yolo26n', details)
    stages = {}
    try:
        if getattr(args, 'pose_device', 'cpu') == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable for RTMW')
            components['club'].set_device('cpu')
            torch.cuda.empty_cache()
            components['model'].to('cuda').float()
            body.metadata.update(inference_device='cuda', precision='float32', execution='staged')
            torch.cuda.synchronize()
            start = time.perf_counter()
            print('Stage 1: RTMW GPU FP32', flush=True)
            try:
                body.replay_results = list(body.predict(video, stream=True))
                torch.cuda.synchronize()
                stages['body_seconds'] = time.perf_counter()-start
            finally:
                components['model'].to('cpu')
                torch.cuda.empty_cache()
            components['club'].set_device(args.club_device)
        start = time.perf_counter()
        print('Stage 2: original full golf pipeline', flush=True)
        sequence = SwingTraceSequence(body, components['club'])
        process(video, sequence, None, components['lifter'],
            handedness=args.handedness, view=args.view, ball_model=components['ball'],
            club_refinement=True, output_folder=folder,
            club_pose_model=components['club'], allow_skeleton_gaps=False,
            frame_overlay=components['club'].draw,
            ball_position=getattr(args, 'ball_position', None),
            club_angle_measurements=({}, {'method': 'swingtrace_box_centres_only',
                                         'missing_direction': 'unobserved'}))
        stages['golf_pipeline_seconds'] = time.perf_counter()-start
    finally:
        body.save_raw(folder)
        components['club'].save(folder)
        write_json(folder/'stage_timing.json', stages)
    manifest = export(folder, video)
    render_wholebody(folder, video, components['model'])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='RTMW body + original complete P1-P10 golf rules')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('video', 'batch', 'check', 'self-test'):
        p = commands.add_parser(name)
        if name == 'video':
            p.add_argument('video', type=Path)
        elif name == 'batch':
            p.add_argument('--input-dir', type=Path, default=ROOT/'pro_refs')
            p.add_argument('--priority', nargs='+', default=[], help='Process these filenames first; still process every video')
        p.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
        from .swingtrace import CHECKPOINT as CLUB_CHECKPOINT
        p.add_argument('--club-checkpoint', type=Path, default=CLUB_CHECKPOINT)
        p.add_argument('--cpu-threads', '--pose-cpu-threads', type=int, default=8)
        p.add_argument('--club-device', choices=['cpu', 'cuda'], default='cpu',
                       help='SwingTrace detector device (FP32)')
        p.add_argument('--pose-device', choices=['cpu', 'cuda'], default='cpu',
                       help='cuda: stage FP32 body predictions before the club pipeline')
        if name == 'batch':
            p.add_argument('--resume', action='store_true', help='Continue an existing batch; preserve interrupted attempts')
        if name not in ('check', 'self-test'):
            p.add_argument('--output-dir', type=Path)
            p.add_argument('--view', choices=['auto', 'face_on', 'down_the_line'], default='auto')
            p.add_argument('--handedness', choices=['auto', 'right', 'left'], default='auto')
            p.add_argument('--golfer-box', nargs=4, type=float)
            p.add_argument('--ball-position', nargs=2, type=float, metavar=('X', 'Y'),
                           help='Ball placement in original pixels; does not imply detected contact')
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.command == 'self-test':
        from .scripts.self_test import run
        return run(args)
    if args.command == 'check':
        from .runtime import load_model
        import cv2
        import numpy as np
        from .backend import extract_prediction
        model = load_model(args.checkpoint, args.cpu_threads).to(args.pose_device)
        from .official import inference
        cap = cv2.VideoCapture(str(ROOT/'pro_refs'/'005.mp4'))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError('Cannot read 005.mp4 for smoke test')
        result = inference(model, frame)[0]
        xy, scores = extract_prediction(result.pred_instances)
        assert xy.shape == (133, 2)
        assert np.isfinite(xy).all() and np.isfinite(scores).all()
        print('PASS: strict checkpoint load and official 133-keypoint single-frame inference')
        print(f'Body confidence: mean={scores[:17].mean():.3f}, min={scores[:17].min():.3f}')
        from .swingtrace import SwingTraceClub
        club = SwingTraceClub(args.club_checkpoint, args.club_device)
        _, confidence, details = club.observe(frame, None, np.full(4, np.nan), 0)
        assert np.isfinite(confidence).all()
        print(f'PASS: SwingTrace detection schema; {details["inference_seconds"]:.3f} seconds')
        return 0
    videos = inputs(args)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_dir or HERE/'outputs'/stamp).resolve()
    resume = getattr(args, 'resume', False)
    if out.exists() and any(out.iterdir()) and not resume:
        parser.error('--output-dir must be empty or new')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(workflow='rtmw', status='initializing', club_device=args.club_device,
                  club_backend='swingtrace-yolo11m', club_checkpoint=str(args.club_checkpoint.resolve()),
                  pose_device=args.pose_device, pose_backend='rtmw-x-2d', pose_checkpoint=str(args.checkpoint.resolve()),
                  precision='float32', input_count=len(videos), results=[])
    if resume:
        import json
        report = json.loads((out/'batch_summary.json').read_text(encoding='utf-8'))
        if report.get('club_backend') != 'swingtrace-yolo11m':
            parser.error('Cannot resume an older club backend into SwingTrace results; use a new output directory')
        if report.get('pose_backend') != 'rtmw-x-2d' or report.get('pose_checkpoint') != str(args.checkpoint.resolve()):
            parser.error('Cannot resume a different body model/checkpoint; use a new output directory')
        completed = {Path(r['video']).resolve() for r in report['results']}
        videos = [v for v in videos if v not in completed]
        report.update(status='initializing', current_video=None)
    overall_started = time.perf_counter()
    write_json(out/'batch_summary.json', report)
    try:
        components = load_components(args)
    except (Exception, SystemExit) as exc:
        report.update(status='initialization_failed', error=str(exc), traceback=traceback.format_exc())
        write_json(out/'batch_summary.json', report)
        print(report['traceback'])
        return 1
    for video in videos:
        folder = out/video.stem
        if folder.exists():
            folder.rename(out/(video.stem+'_interrupted_'+stamp))
        folder.mkdir()
        started = time.perf_counter()
        record = dict(video=str(video), output=str(folder))
        report.update(status='running', current_video=video.name)
        write_json(out/'batch_summary.json', report)
        print(f'Processing {video.name}', flush=True)
        try:
            manifest = run_video(video, folder, components, args)
            record.update(status='success', frames=manifest['frames'])
        except (Exception, SystemExit) as exc:
            record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
            print(record['traceback'], flush=True)
        record['seconds'] = round(time.perf_counter()-started, 2)
        write_json(folder/'run_status.json', record)
        report['results'].append(record)
        write_json(out/'batch_summary.json', report)
    report.update(status='completed', current_video=None,
                  invocation_seconds=round(time.perf_counter()-overall_started, 2),
                  success=sum(r['status']=='success' for r in report['results']),
                  failed=sum(r['status']=='failed' for r in report['results']))
    write_json(out/'batch_summary.json', report)
    from .scripts.audit_results import audit
    validation = audit(out)
    report['artifact_issues'] = sum(len(r['artifact_issues']) for r in validation['results'])
    write_json(out/'batch_summary.json', report)
    print(f'Results: {out}', flush=True)
    return 2 if report['failed'] or report['artifact_issues'] else 0
