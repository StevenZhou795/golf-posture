"""The three public workflow commands and their saved data contracts."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs' / 'workflows'
POSITIONS = [f'p{i}' for i in range(1, 11)]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def new_output(requested, default):
    out = Path(requested or default).resolve()
    if out.exists() and any(out.iterdir()):
        if requested:
            raise ValueError(f'Output folder must be empty: {out}')
        out = out.with_name(out.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=True)
    return out


def publish(stage, out, runs_root=None):
    runs_root = Path(runs_root) if runs_root is not None else RUNS
    runs_root.mkdir(parents=True, exist_ok=True)
    write_json(runs_root / f'latest_{stage}.json', {'folder': str(out)})


def latest(stage, runs_root=None):
    pointer = (Path(runs_root) if runs_root is not None else RUNS) / f'latest_{stage}.json'
    if not pointer.is_file():
        raise ValueError(f'No completed {stage} workflow. Run that command first or supply its folder.')
    return Path(read_json(pointer)['folder'])


def validate_reference_set(folder):
    import cv2
    from golf_workflows.standard_frames import position_key
    entries = {}
    for path in Path(folder).iterdir():
        if not path.is_dir() or not (path / 'joints.npz').is_file():
            continue
        key = position_key(path)
        if key not in POSITIONS:
            continue
        if key in entries:
            raise ValueError(f'Duplicate reference position: {key}')
        with np.load(path / 'joints.npz') as data:
            pose = data['reference_3d']
            if pose.ndim != 2 or pose.shape[1] != 3 or pose.shape[0] not in (19,22) or not np.isfinite(pose).all():
                raise ValueError(f'Invalid reference coordinates: {path}')
        if cv2.imread(str(path / 'reference_3d.jpg')) is None:
            raise ValueError(f'Missing reference image: {path}')
        entries[key] = {'folder': path.name, 'image': f'{path.name}/reference_3d.jpg',
                        'coordinates': f'{path.name}/joints.npz'}
    if set(entries) != set(POSITIONS):
        raise ValueError(f'Reference set must contain P1–P10; missing {sorted(set(POSITIONS)-set(entries))}')
    return {key: entries[key] for key in POSITIONS}


def references(args):
    from golf_workflows import standard_frames
    runs_root = getattr(args, 'runs_root', RUNS)
    sheets = sorted(args.input_dir.glob('*.jpg'))
    keys = [standard_frames.position_key(path) for path in sheets]
    if len(keys) != 10 or set(keys) != set(POSITIONS):
        raise ValueError('Input must contain exactly one P1–P10 JPG sheet each')
    cached = validate_reference_set(args.from_run) if args.from_run else None
    out = new_output(args.output_dir, runs_root / 'references')
    if cached:
        for entry in cached.values():
            shutil.copytree(args.from_run / entry['folder'], out / entry['folder'])
    else:
        standard_frames.main(['--input-dir', str(args.input_dir), '--output-dir', str(out),
                              '--pose-backend', args.pose_backend, '--lifter', args.lifter,
                              '--club-pose-backend', getattr(args, 'club_pose_backend', 'vitpose-huge'),
                              '--club-crop-backend', getattr(args, 'club_crop_backend', 'official-yolox-2cls'),
                              '--pose-cpu-threads', str(args.pose_cpu_threads)])
    entries = validate_reference_set(out)
    from golf_workflows.reference_previews import create_reference_previews
    previews = create_reference_previews(out) if cached else {
        'body': 'reference_3d_preview.png', 'club_overlay': 'reference_3d_club_overlay_preview.png',
        'combined': 'reference_all_preview.png'}
    model_summary = args.from_run / 'run_summary.json' if cached else out / 'run_summary.json'
    if cached and not model_summary.is_file():
        model_summary = args.from_run / 'workflow.json'
    manifest = dict(schema_version=1, workflow='references', input_dir=str(args.input_dir.resolve()),
                    from_run=str(args.from_run.resolve()) if cached else None,
                    fresh_2d_inference=not bool(cached), positions=entries,
                    models=read_json(model_summary).get('models') if model_summary.is_file() else None,
                    numerical_input='joints.npz:reference_3d', image_count=10, previews=previews)
    write_json(out / 'workflow.json', manifest)
    publish('references', out, runs_root)
    print(f'10 reference images and coordinate sets: {out}')


def export_identified_frames(folder):
    """Save exactly ten selected poses; scoring need not select frames again."""
    from golf_workflows.score_swing import frame_of
    from golf_workflows.standard_frames import save_still
    arrays = sorted(folder.glob('*_3d.npz'))
    if len(arrays) != 1:
        raise ValueError('Video run must contain exactly one *_3d.npz file')
    summary_path = arrays[0].with_suffix('.json')
    summary = read_json(summary_path)
    indices = [frame_of(summary, key) for key in POSITIONS]
    with np.load(arrays[0]) as data:
        poses = data['keypoints_3d']
        if any(f is None or not 0 <= f < len(poses) for f in indices):
            raise ValueError('Frame identification did not produce all ten valid indices; supply --positions-json for reviewed indices')
        if any(a >= b for a,b in zip(indices, indices[1:])):
            raise ValueError('P1–P10 indices must be distinct and increasing; supply --positions-json for reviewed indices')
        selected = poses[indices]
        joints = data['joints']
        bundle = {}
        for source, target in (
            ('keypoints_2d', 'keypoints_2d'),
            ('keypoints_2d_valid', 'keypoints_2d_valid'),
            ('keypoints_2d_filled', 'keypoints_2d_filled'),
            ('keypoints_2d_normalized', 'keypoints_2d_normalized'),
            ('club_landmark_confidence', 'club_landmark_confidence'),
            ('keypoints_2d_confidence', 'keypoints_2d_confidence'),
            ('club_evidence', 'club_evidence'),
        ):
            if source in data:
                bundle[target] = data[source][indices]
        if 'club_evidence' not in bundle and 'keypoints_2d' in bundle and 'keypoints_2d_valid' in bundle:
            from types import SimpleNamespace
            from golf_workflows.golf_pose_3d import video_club_evidence
            evidence = video_club_evidence(SimpleNamespace(frames=len(indices),
                points=bundle['keypoints_2d'], valid=bundle['keypoints_2d_valid'],
                keypoint_confidence=bundle.get('keypoints_2d_confidence'),
                club_confidence=bundle.get('club_landmark_confidence')))
            bundle['club_evidence'] = evidence
    if not np.isfinite(selected).all():
        raise ValueError('One or more selected poses lack finite 3D coordinates')
    for key, pose in zip(POSITIONS, selected):
        if not (folder / f'{key.upper()}.jpg').is_file():
            raise ValueError(f'Missing selected source image: {key}')
        evidence = str(bundle['club_evidence'][POSITIONS.index(key)]) if 'club_evidence' in bundle else None
        title = key.upper() + (' (club inferred)' if evidence == 'shaft_fallback' else '')
        save_still(pose, folder / f'{key.upper()}_3d.jpg', title, show_club=evidence != 'none')
    np.savez_compressed(folder / 'identified_frames.npz', positions=np.array(POSITIONS),
                        source_frame_indices=np.array(indices), joints=joints, keypoints_3d=selected,
                        **bundle)
    manifest = dict(schema_version=1, workflow='video', source=summary['source'], frame_index_base=0,
                    lead_side=summary.get('lead_side'), handedness=summary.get('handedness'),
                    view=summary.get('view'), club_tracking=summary.get('club_tracking'),
                    checkpoint=summary.get('checkpoint'), pose_model=summary.get('pose_model'),
                    pose_backend=summary.get('pose_backend'),
                    club_refinement=summary.get('club_refinement'),
                    checkpoint_selection=summary.get('checkpoint_selection'),
                    frames=[dict(position=key, frame_index=frame, image=f'{key.upper()}.jpg',
                                 pose_image=f'{key.upper()}_3d.jpg') for key,frame in zip(POSITIONS,indices)],
                    coordinates='identified_frames.npz')
    if 'club_landmark_confidence' in bundle:
        manifest['club_landmarks'] = [str(name) for name in joints[17:]]
        manifest['club_landmark_confidence'] = 'identified_frames.npz:club_landmark_confidence'
    if 'keypoints_2d_confidence' in bundle:
        manifest['keypoint_confidence'] = 'identified_frames.npz:keypoints_2d_confidence'
    if 'club_evidence' in bundle:
        for row, evidence in zip(manifest['frames'], bundle['club_evidence']):
            row['club_evidence'] = str(evidence)
    write_json(folder / 'identified_frames.json', manifest)
    return manifest


def video(args):
    runs_root = getattr(args, 'runs_root', RUNS)
    golfer_box = getattr(args, 'golfer_box', None)
    no_auto_box = getattr(args, 'no_auto_golfer_box', False)
    if golfer_box is not None or no_auto_box:
        if args.from_run:
            raise ValueError('--golfer-box/--no-auto-golfer-box require fresh video inference, not --from-run')
    if golfer_box is not None:
        if not (args.pose_backend == 'golfpose-golfer-vitpose-huge'
                or (args.pose_backend == 'yolo26n-pose'
                    and getattr(args, 'club_crop_backend', 'official-yolox-2cls') == 'official-yolox-2cls')):
            raise ValueError('--golfer-box applies to the joint GolfPose ViTPose backend, '
                             'or yolo26n-pose with --club-crop-backend official-yolox-2cls')
        from golf_workflows.golfer_vitpose import validate_golfer_box
        golfer_box = validate_golfer_box(golfer_box)
    if args.from_run and args.video:
        raise ValueError('Supply a video or --from-run, not both')
    if not args.from_run and not args.video:
        raise ValueError('Supply an input video')
    if args.club_marker_length_m is not None and (args.no_club_refinement or not np.isfinite(args.club_marker_length_m) or args.club_marker_length_m <= 0):
        raise ValueError('Measured marker distance must be positive and requires club refinement')
    if args.from_run:
        pairs = sorted(args.from_run.glob('*_3d.npz'))
        if len(pairs) != 1 or not pairs[0].with_suffix('.json').is_file():
            raise ValueError('Cached run requires one matching *_3d.npz/json pair')
        stem = pairs[0].stem.removesuffix('_3d')
    else:
        if not args.video.is_file():
            raise ValueError(f'Video not found: {args.video}')
        stem = args.video.stem
    out = new_output(args.output_dir, runs_root / 'videos' / stem)
    if args.from_run:
        if args.positions_json or args.club_marker_length_m is not None or args.no_club_refinement:
            raise ValueError('--from-run reuses completed results; refinement/selection overrides require a fresh video run')
        for path in args.from_run.iterdir():
            if path.is_file() and path.suffix.lower() in ('.npz','.json','.jpg','.avi','.mp4'):
                shutil.copy2(path, out / path.name)
        summary_path = out / pairs[0].with_suffix('.json').name
        summary = read_json(summary_path)
        summary['checkpoint_selection'] = {**summary.get('checkpoint_selection',{}),'fresh_2d_inference': False}
        summary['imported_from_run'] = str(args.from_run.resolve())
        # Update artifact links; imported numerical arrays stay byte-identical.
        for key in ('video','still','arrays'):
            if summary.get(key):
                summary[key] = str(out / Path(summary[key]).name)
        write_json(summary_path, summary)
    else:
        from golf_workflows import golf_pose_3d
        command = [str(args.video.resolve()), '--run-dir', str(out),
                   '--pose-backend', args.pose_backend, '--lifter', args.lifter,
                   '--club-pose-backend', getattr(args, 'club_pose_backend', 'vitpose-huge'),
                   '--club-crop-backend', getattr(args, 'club_crop_backend', 'official-yolox-2cls'),
                   '--pose-cpu-threads', str(args.pose_cpu_threads),
                   '--handedness', args.handedness, '--view', args.view]
        if not args.no_club_refinement:
            command.append('--club-refinement')
        if getattr(args, 'track_club_path', False):
            command.append('--track-club-path')
        if args.club_marker_length_m is not None:
            command += ['--club-marker-length-m', str(args.club_marker_length_m)]
        if args.positions_json:
            command += ['--positions-json', str(args.positions_json.resolve())]
        if golfer_box is not None:
            command += ['--golfer-box', *map(str, golfer_box)]
        if no_auto_box:
            command.append('--no-auto-golfer-box')
        if getattr(args, 'no_club_mask', False):
            command.append('--no-club-mask')
        if getattr(args, 'no_body_fusion', False):
            command.append('--no-body-fusion')
        if getattr(args, 'allow_skeleton_gaps', False):
            command.append('--allow-skeleton-gaps')
        golf_pose_3d.main(command)
    manifest = export_identified_frames(out)
    manifest['from_run'] = str(args.from_run.resolve()) if args.from_run else None
    write_json(out / 'workflow.json', manifest)
    publish('video', out, runs_root)
    print(f'10 identified source frames, 3D images and poses: {out}')


def load_identified_frames(folder):
    from golf_workflows.golf_pose_3d import joint_names
    manifest = read_json(folder / 'identified_frames.json')
    with np.load(folder / 'identified_frames.npz') as data:
        if list(data['positions']) != POSITIONS or data['keypoints_3d'].shape[0] != 10:
            raise ValueError('Selected pose bundle must contain P1–P10 in order')
        poses = data['keypoints_3d'].copy()
        source_indices = data['source_frame_indices'].tolist()
        joints = data['joints'].tolist()
    if poses.ndim != 3 or poses.shape[1] not in (19,22) or poses.shape[2] != 3 or not np.isfinite(poses).all():
        raise ValueError('Selected pose coordinates must be finite 19/22-joint XYZ arrays')
    if joints != joint_names(poses.shape[1]):
        raise ValueError('Selected joint order does not match GolfPose')
    if len(source_indices) != 10 or any(type(i) is not int or i < 0 for i in source_indices) or any(a >= b for a,b in zip(source_indices,source_indices[1:])):
        raise ValueError('Selected source indices must be nonnegative, distinct and increasing')
    records = manifest.get('frames', [])
    if [row['position'] for row in records] != POSITIONS or [row['frame_index'] for row in records] != source_indices:
        raise ValueError('Selected frame manifest and coordinate bundle disagree')
    summary = dict(positions={key:i for i,key in enumerate(POSITIONS)},
                   source_frame_indices=dict(zip(POSITIONS,source_indices)),
                   lead_side=manifest.get('lead_side'), handedness=manifest.get('handedness'))
    return dict(folder=folder, summary=summary, poses=poses)


def score(args):
    from golf_workflows.compare_standard_frames import compare, POSTURE
    runs_root = getattr(args, 'runs_root', RUNS)
    user = (args.run or latest('video', runs_root)).resolve()
    refs = (args.references or latest('references', runs_root)).resolve()
    validate_reference_set(refs)
    selected = load_identified_frames(user)
    out = new_output(args.output_dir, runs_root / 'scores' / user.name)
    result = compare(user, standard_dir=refs, output_dir=out, selected_swing=selected)
    result.update(schema_version=1, workflow='score', video_run=str(user), references=str(refs),
                  numerical_input=str(user / 'identified_frames.npz'), criteria=list(POSTURE),
                  methodology=str(ROOT / 'SCORING_METHODOLOGY.md'),
                  note='Scores use saved joint coordinates; JPG images are visualizations')
    result['coverage'] = {'positions_scored': sum(value is not None for value in result['positions'].values()),
                          'parameters_scored': sum(len(value['parameters']) for value in result['positions'].values() if value)}
    write_json(out / 'score.json', result)
    write_json(out / 'workflow.json', {'workflow':'score','video_run':str(user),'references':str(refs),'score':'score.json'})
    publish('score', out, runs_root)
    print(f'Score: {result["pose_score"]}/100; output: {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Three golf workflows: references, video, score.')
    sub = parser.add_subparsers(dest='workflow',required=True)
    ref = sub.add_parser('references',help='Build ten reference_3d.jpg images and coordinate sets')
    ref.add_argument('--input-dir',type=Path,default=ROOT/'pro_refs'/'Standard_10_frames')
    ref.add_argument('--from-run',type=Path,help='Reuse an existing completed reference set')
    vid = sub.add_parser('video',help='Identify and export P1–P10 from a swing video')
    vid.add_argument('video',type=Path,nargs='?')
    vid.add_argument('--from-run',type=Path,help='Export ten poses from an existing completed video run')
    vid.add_argument('--positions-json',type=Path,help='Optional reviewed P1–P10 selection')
    vid.add_argument('--golfer-box',type=float,nargs=4,metavar=('X1','Y1','X2','Y2'),
                     help='Golfer ROI used where the GolfPose detector misses; overrides the automatic YOLO26n ROI')
    vid.add_argument('--no-auto-golfer-box',action='store_true',
                     help='Do not derive a default golfer ROI from a YOLO26n pass over sampled frames')
    vid.add_argument('--no-club-refinement',action='store_true')
    vid.add_argument('--track-club-path',action='store_true',
                     help='Opt in to temporal club tracking and its rendered path')
    vid.add_argument('--club-marker-length-m',type=float)
    vid.add_argument('--handedness',choices=('auto','right','left'),default='auto',
                     help='auto: from the 2D hands; falls back to right when ambiguous')
    vid.add_argument('--view',choices=('auto','down_the_line','face_on'),default='auto',
                     help='auto: from shoulder width and the feet line; falls back to down_the_line')
    vid.add_argument('--no-club-mask',action='store_true',
                     help='Do not use the golf-club segmentation mask as a shaft-angle measurement')
    vid.add_argument('--allow-skeleton-gaps',action='store_true',
                     help='Warn instead of aborting when the golfer skeleton is lost inside the swing window')
    scoring = sub.add_parser('score',help='Score the exported ten poses against ten references')
    scoring.add_argument('run',type=Path,nargs='?',help='Video workflow folder; defaults to latest completed video')
    scoring.add_argument('--references',type=Path,help='Reference workflow folder; defaults to latest completed references')
    for command in (ref,vid):
        command.add_argument('--pose-backend',choices=('yolo26n-pose','vitpose-huge','golfpose-golfer-vitpose-huge'),default='yolo26n-pose')
        command.add_argument('--lifter',choices=('17+2','17+5'),default='17+2')
        command.add_argument('--club-pose-backend',choices=('segmentation','vitpose-huge'),default='vitpose-huge')
        command.add_argument('--club-crop-backend',
                             choices=(('golfer-box','segmentation','official-yolox-2cls') if command is ref
                                      else ('segmentation','official-yolox-2cls')),
                             default='official-yolox-2cls')
        command.add_argument('--pose-cpu-threads',type=int,default=8)
    for command in (ref,vid,scoring):
        command.add_argument('--output-dir',type=Path)
    args = parser.parse_args(argv)
    try:
        {'references':references,'video':video,'score':score}[args.workflow](args)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        parser.error(str(exc))
