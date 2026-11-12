import json
from pathlib import Path
import cv2
import numpy as np


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding='utf-8')


def selection_source(position, summary):
    selection = summary.get('checkpoint_selection', {})
    explicit = (selection.get('position_sources') or {}).get(position)
    if explicit:
        return explicit
    phase_key = {'p1': 'address_source', 'p4': 'top_source', 'p7': 'impact_source'}.get(position)
    if phase_key:
        return summary.get('phase_detection', {}).get(phase_key, 'unspecified_phase_evidence')
    if selection.get('selector') == 'joint_dp':
        return {'p3': 'lead_arm_horizontal_rising', 'p5': 'lead_arm_horizontal_falling',
                'p9': 'follow_through_arm_horizontal', 'p10': 'raised_hands_nearest_head'}.get(position, 'joint_dp')
    return selection.get('selector') or selection.get('method') or 'original_rule_based'


def export(folder, video):
    from golf_workflows.cli import export_identified_frames
    summary = json.loads(next(folder.glob('*_3d.json')).read_text(encoding='utf-8'))
    fallback = summary.get('phase_detection', {}).get('fallback')
    if fallback == 'no_corroborated_wrist_rise_and_return':
        raise ValueError('Untrusted P1-P10: original detector reports ' + fallback)
    manifest = export_identified_frames(folder)
    with np.load(folder / 'wholebody.npz') as data:
        times = data['times_ms'].copy()
    selection = summary['checkpoint_selection']
    for row in manifest['frames']:
        p = row['position']
        row.update(time_ms=float(times[row['frame_index']]),
            confidence=(selection.get('position_confidence') or {}).get(p),
            source=selection_source(p, summary),
            subframe_ms=(selection.get('subframe_ms') or {}).get(p))
        row['confidence_note'] = ('Original selector score; not a calibrated probability.' if row['confidence'] is not None
                                  else 'Original phase detector does not emit a numerical confidence for this anchor.')
    native = folder/'swingtrace_detections.json'
    if native.is_file():
        from .frame_evidence import enrich_manifest
        observations = json.loads(native.read_text(encoding='utf-8'))
        enrich_manifest(manifest, observations['frames'], summary)
        write_json(folder/'frame_evidence.json', dict(model=observations['model'], frames=manifest['frames']))
    write_json(folder / 'identified_frames.json', manifest)
    from .runtime import ROOT
    benchmark = ROOT/'benchmarks/p_position_labels.json'
    if benchmark.is_file():
        clip = json.loads(benchmark.read_text(encoding='utf-8')).get('clips', {}).get(video.stem, {})
        labels = clip.get('labels', {})
        rows = []
        for row in manifest['frames']:
            interval = labels.get(row['position'])
            if interval is not None:
                frame = row['frame_index']
                lo, hi = interval
                rows.append(dict(position=row['position'], predicted_frame=frame, label_range=interval,
                    signed_error_frames=frame-lo if frame < lo else frame-hi if frame > hi else 0))
        write_json(folder/'label_comparison.json', dict(source=str(benchmark),
            note='Partial reviewed labels; unlabelled positions are not evaluated.', positions=rows))
    # Consistent, readable 5 x 2 contact sheet of the shared pipeline stills.
    tiles = []
    for i in range(1, 11):
        image = cv2.imread(str(folder / f'P{i}.jpg'))
        if image is None:
            raise ValueError(f'Missing P{i} still')
        ratio = min(320/image.shape[1], 300/image.shape[0])
        image = cv2.resize(image, (round(image.shape[1]*ratio), round(image.shape[0]*ratio)))
        tile = np.full((336, 320, 3), 245, np.uint8)
        tile[36:36+image.shape[0], (320-image.shape[1])//2:(320+image.shape[1])//2] = image
        cv2.putText(tile, f'P{i}  frame {manifest["frames"][i-1]["frame_index"]}',
            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 30), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not cv2.imwrite(str(folder / 'ten_frames.jpg'), np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:])])):
        raise OSError('Failed to write contact sheet')
    return manifest


def render_wholebody(folder, video, model):
    from .overlay import draw_detections
    native = folder/'swingtrace_detections.json'
    observations = ({r['frame_index']: r for r in json.loads(native.read_text(encoding='utf-8'))['frames']}
                    if native.is_file() else {})
    with np.load(folder / 'wholebody.npz') as data:
        xy, scores = data['image_2d'], data['confidence']
    links = model.dataset_meta.get('skeleton_links', [])
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(str(folder / 'rtmw_2d.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        capture.release()
        raise OSError('Cannot create rtmw_2d.mp4')
    try:
        for index, points in enumerate(xy):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Source video ended before saved predictions')
            good = (scores[index] >= .3) & np.isfinite(points).all(axis=1)
            for a, b in links:
                if good[a] and good[b]:
                    cv2.line(frame, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (80, 220, 40), 2, cv2.LINE_AA)
            for p in points[good]:
                cv2.circle(frame, tuple(np.rint(p).astype(int)), 2, (0, 180, 255), -1)
            draw_detections(frame, observations.get(index))
            cv2.putText(frame, f'RTMW-X 2D  frame {index}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
