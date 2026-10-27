"""Audit cached club tracking; never modifies source runs or workflow defaults."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/golf_pose_3d_vitpose_huge_17plus5/b85ed0b4db566e63bce190824c2f2196'


def angle(u, v):
    den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(u*v, axis=-1) / np.maximum(den, 1e-12), -1, 1)))


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), max=float(x.max()))


def measure(poses, smoothed, valid):
    vector = poses[:,18] - poses[:,17]
    length = np.linalg.norm(vector, axis=-1)
    finite = np.isfinite(length)
    both = finite & valid[:,17] & valid[:,18]
    discrepancy = angle(vector[:,:2], smoothed[:,18] - smoothed[:,17])
    return dict(length_all_lifted_m=stats(length), length_detected_endpoints_m=stats(length[both]),
                projected_axis_disagreement_deg=stats(discrepancy[both]),
                angular_change_per_frame_deg=stats(angle(vector[1:], vector[:-1]))), length, discrepancy


def main():
    data = np.load(next(SOURCE.glob('*_3d.npz')))
    poses, sm, valid = (data[k] for k in ['keypoints_3d', 'keypoints_2d_filled', 'keypoints_2d_valid'])
    raw = data['keypoints_2d']
    finite = np.isfinite(poses).all(axis=(1,2))
    frames = np.flatnonzero(finite)
    both = finite & valid[:,17] & valid[:,18]
    latest, length, discrepancy = measure(poses, sm, valid)
    ratio = np.linalg.norm(sm[:,18]-sm[:,17], axis=-1) / np.linalg.norm(raw[:,18]-raw[:,17], axis=-1)
    missing = frames[~both[frames]]
    gaps = np.split(missing, np.flatnonzero(np.diff(missing)>1)+1)
    report = dict(source=str(SOURCE.relative_to(ROOT)), frame_index_base=0,
                  lifted_frames=len(frames), both_endpoints_detected=int(both.sum()),
                  missing_endpoint_frames=int(len(frames)-both.sum()),
                  longest_missing_endpoint_run=max(len(g) for g in gaps),
                  latest_17plus5=latest, smoothed_to_detected_2d_length_ratio=stats(ratio[both]),
                  shortest_frame=int(np.nanargmin(length)), longest_frame=int(np.nanargmax(length)),
                  limitations=['No ground-truth 3D, camera calibration, or measured club-marker distance.',
                               'Projected-axis discrepancy compares camera XY axes under an orthographic approximation; it is not calibrated reprojection error.',
                               'Raw 2D points already include tracker endpoint blending.',
                               'Checkpoint comparison measures consistency only, not physical accuracy.'])
    # Controlled comparison: same body, club detections, smoothing, normalization and windows.
    import torch
    import golf_pose_3d as workflow
    torch.set_num_threads(8)
    model = workflow.load_lifter(workflow.CHECKPOINT)
    alternative = np.full((len(poses),19,3), np.nan, dtype=np.float32)
    alternative[frames[0]:frames[-1]+1] = workflow.lift_sequence(
        model, data['keypoints_2d_normalized'][frames[0]:frames[-1]+1,:19])
    comparison, alt_length, alt_discrepancy = measure(alternative, sm, valid)
    report['controlled_17plus2'] = comparison
    np.savez_compressed(OUT/'controlled_17plus2.npz', keypoints_3d=alternative)
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savetxt(OUT/'per_frame.csv', np.column_stack([np.arange(len(poses)), finite, both, length,
               alt_length, discrepancy, alt_discrepancy, ratio]), delimiter=',',
               header='frame,lifted,both_endpoints_detected,length_22_m,length_19_m,axis_disagreement_22_deg,axis_disagreement_19_deg,smooth_raw_2d_length_ratio', comments='')
    fig, axes = plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].plot(length,label='17+5 (three imputed club inputs)')
    axes[0].plot(alt_length,label='17+2, identical 2D input',alpha=.8)
    axes[0].set_ylabel('Predicted marker distance (m)')
    axes[0].legend()
    axes[1].plot(np.where(both,discrepancy,np.nan),label='17+5')
    axes[1].plot(np.where(both,alt_discrepancy,np.nan),label='17+2',alpha=.7)
    axes[1].set_ylabel('XY axis disagreement (deg)')
    axes[2].plot(np.where(both,ratio,np.nan))
    axes[2].axhline(1,color='black',lw=.7)
    axes[2].set_ylabel('Smoothed / detected 2D length')
    axes[2].set_xlabel('Video frame (zero based)')
    for ax in axes:
        ax.set_xlim(frames[0],frames[-1])
        ax.grid(alpha=.2)
        for gap in gaps:
            if len(gap): ax.axvspan(gap[0]-.5,gap[-1]+.5,color='gray',alpha=.12)
        ax.axvline(303,color='red',linestyle=':',alpha=.7)
    fig.suptitle('Club tracking audit — gray: missing endpoint; red: model-window handoff\nConsistency diagnostics; no ground-truth 3D accuracy measurement')
    fig.tight_layout()
    fig.savefig(OUT/'shaft_diagnostics.png',dpi=160)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
