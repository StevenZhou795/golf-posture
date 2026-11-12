"""Check exported batch artifacts and aggregate partial-label frame errors."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rtmw_workflow.report import write_json


def audit(root):
    batch = json.loads((root/'batch_summary.json').read_text(encoding='utf-8'))
    rows, errors = [], []
    for result in batch['results']:
        folder = Path(result['output'])
        row = dict(video=Path(result['video']).name, status=result['status'], seconds=result['seconds'])
        issues = []
        if (folder/'wholebody.npz').is_file():
            with np.load(folder/'wholebody.npz') as data:
                row['decoded_frames'] = len(data['frame_index'])
                row['mean_body_confidence'] = float(data['confidence'][:, :17].mean()) if row['decoded_frames'] else None
                if data['image_2d'].shape != (row['decoded_frames'], 133, 2):
                    issues.append('Unexpected wholebody shape')
                if not np.all(np.diff(data['times_ms']) > 0):
                    issues.append('Nonmonotonic timestamps')
                if not np.isfinite(data['times_ms']).all() or np.any(data['times_ms'] < 0):
                    issues.append('Invalid or negative timestamps')
        if result['status'] == 'success':
            summary = json.loads(next(folder.glob('*_3d.json')).read_text(encoding='utf-8'))
            fallback = summary.get('phase_detection', {}).get('fallback')
            if fallback == 'no_corroborated_wrist_rise_and_return':
                issues.append('Untrusted P1-P10: ' + fallback)
            manifest = json.loads((folder/'identified_frames.json').read_text(encoding='utf-8'))
            frames = [r['frame_index'] for r in manifest['frames']]
            if len(frames) != 10 or not all(a < b for a, b in zip(frames, frames[1:])):
                issues.append('Invalid P1-P10 order/count')
            for i in range(1, 11):
                if cv2.imread(str(folder/f'P{i}.jpg')) is None:
                    issues.append(f'Unreadable P{i}.jpg')
            capture = cv2.VideoCapture(str(folder/'rtmw_2d.mp4'))
            row['overlay_frames'] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if row['overlay_frames'] != row.get('decoded_frames'):
                issues.append('Overlay/raw frame count mismatch')
            if (folder/'label_comparison.json').is_file():
                comparison = json.loads((folder/'label_comparison.json').read_text(encoding='utf-8'))
                row['label_comparison'] = comparison['positions']
                if not issues:
                    errors.extend(abs(r['signed_error_frames']) for r in comparison['positions'])
        else:
            row['error'] = result.get('error')
        row['artifact_issues'] = issues
        row['validation_status'] = 'passed' if result['status'] == 'success' and not issues else 'failed'
        rows.append(row)
    report = dict(batch_status=batch.get('status'), input_count=batch.get('input_count'),
        completed_count=len(rows), results=rows, labelled_positions=len(errors),
        mean_absolute_distance_to_label_range=float(np.mean(errors)) if errors else None,
        total_seconds=sum(r['seconds'] for r in rows),
        note='Partial labels only; execution or validation failures do not contribute to frame error statistics.')
    write_json(root/'validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.batch_folder), ensure_ascii=False, indent=2))
