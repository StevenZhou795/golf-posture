"""Replay only the lifting/refinement stage from an existing workflow run.

Usage: python replay_club_refinement.py SOURCE_RUN --output-dir NEW_RUN
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from golf_workflows import golf_pose_3d as workflow
from golf_workflows.club_refinement import prepare_club_inputs, refine_club_3d
from golf_workflows.swing_phases import SwingPhases
from golf_workflows.trace_swing_plane import ClubSample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--club-marker-length-m',type=float)
    args=parser.parse_args()
    source=args.source.resolve()
    out=args.output_dir.resolve()
    if out==source or source in out.parents:
        parser.error('Output must be separate from the source run')
    if out.exists() and any(out.iterdir()):
        parser.error('Output folder must be empty')
    files=sorted(p for p in source.rglob('*') if p.is_file())
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    npz=next(source.glob('*_3d.npz'))
    meta=json.loads(npz.with_suffix('.json').read_text(encoding='utf-8'))
    d=np.load(npz)
    raw, valid=d['keypoints_2d'],d['keypoints_2d_valid']
    confidence=d['club_confidence'] if 'club_confidence' in d else np.ones(len(raw))
    confidence_source=meta.get('confidence_source','saved_detector_confidence') if 'club_confidence' in d else 'binary_validity_only_original_confidence_unavailable'
    prepared,_=prepare_club_inputs(raw,d['keypoints_2d_filled'],valid,confidence)
    # Recover the exact old affine from cached normalized/body coordinates,
    # avoiding rounded center/scale values in the JSON summary.
    oldfilled=d['keypoints_2d_filled']
    normalized=d['keypoints_2d_normalized'].copy()
    x=oldfilled[:,:17].reshape(-1,2).astype(float)
    y=normalized[:,:17].reshape(-1,2).astype(float)
    x=x-x.mean(axis=0); y=y-y.mean(axis=0)
    scale=float(np.sum(x*y)/np.sum(x**2))
    # Translation differs by axis; only the common scale is required for deltas.
    normalized[:,17:]=d['keypoints_2d_normalized'][:,17:]+(prepared[:,17:]-oldfilled[:,17:])*scale
    torch.set_num_threads(8)
    model=workflow.load_lifter(workflow.ROOT/meta['checkpoint'])
    start,stop=meta['lift_window']
    before=np.full_like(d['keypoints_3d'],np.nan)
    before[start:stop+1]=workflow.lift_sequence(model,normalized[start:stop+1],blend_windows=True)
    poses,refinement=refine_club_3d(before,normalized,valid,confidence,args.club_marker_length_m)
    out.mkdir(parents=True,exist_ok=True)
    arrays={key:d[key] for key in d.files}
    arrays.update(keypoints_2d_filled=prepared,keypoints_2d_normalized=normalized,
                  keypoints_3d=poses,keypoints_3d_before_club_refinement=before,club_confidence=confidence)
    np.savez_compressed(out/npz.name,**arrays)
    samples=[]
    for f in np.flatnonzero(valid[:,18]):
        samples.append(ClubSample(int(f),raw[f,17].copy() if valid[f,17] else None,
                                  raw[f,18].copy(),float(confidence[f]),bool(samples and f-samples[-1].frame<=6)))
    swing=workflow.Swing2D(len(raw),float(d['fps']),tuple(d['frame_size']),raw,valid,prepared,
                          np.full((len(raw),4),np.nan),samples,np.zeros(len(raw)))
    phases=SwingPhases(**meta['phases'])
    video=workflow.ROOT/meta['source']
    avi=out/(video.stem+'_3d.avi')
    still=out/(video.stem+'_3d.jpg')
    written=workflow.render_video(video,swing,poses,avi,still,phases,meta['ball']['p1_xy'])
    stills=workflow.save_position_stills(video,swing,meta['positions'],out,phases,meta['ball']['p1_xy'])
    def metrics(p):
        vec=p[:,18]-p[:,17]
        length=np.linalg.norm(vec,axis=1)
        good=np.isfinite(length)&valid[:,17]&valid[:,18]
        target=raw[:,18]-raw[:,17]
        angle=np.degrees(np.arccos(np.clip(np.sum(vec[:,:2]*target,axis=1)/
                          np.maximum(np.linalg.norm(vec[:,:2],axis=1)*np.linalg.norm(target,axis=1),1e-8),-1,1)))
        b3=p[:,:17,:2]-p[:,:17,:2].mean(axis=1,keepdims=True)
        b2=normalized[:,:17]-normalized[:,:17].mean(axis=1,keepdims=True)
        projection=np.sum(b3*b2,axis=(1,2))/np.maximum(np.sum(b3*b3,axis=(1,2)),1e-8)
        median_projection=np.nanmedian(projection[projection>0])
        projection=np.clip(projection,.5*median_projection,2*median_projection)
        target_length=np.linalg.norm(target,axis=1)*scale
        relative_length_error=np.abs(np.linalg.norm(vec[:,:2],axis=1)*projection-target_length)/np.maximum(target_length,1e-8)
        return dict(length_mean_m=float(np.nanmean(length)),length_std_m=float(np.nanstd(length)),
                    length_min_m=float(np.nanmin(length)),length_max_m=float(np.nanmax(length)),
                    detected_raw_axis_median_deg=float(np.median(angle[good])),
                    detected_raw_axis_p90_deg=float(np.percentile(angle[good],90)),
                    weak_perspective_projected_length_relative_error_median=float(np.median(relative_length_error[good])),
                    weak_perspective_projected_length_relative_error_p90=float(np.percentile(relative_length_error[good],90)))
    validation=dict(original=metrics(d['keypoints_3d']),new_lifting_before_constraint=metrics(before),
                    refined=metrics(poses),confidence_source=confidence_source,refinement=refinement)
    assert np.array_equal(poses[:,:18],before[:,:18],equal_nan=True)
    assert np.array_equal(valid,arrays['keypoints_2d_valid'])
    cap=cv2.VideoCapture(str(avi)); decoded=0
    while cap.read()[0]: decoded+=1
    cap.release()
    assert written==decoded==len(raw)
    for name,digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    validation.update(frames_written=written,frames_decoded=decoded,source_files_unchanged=len(hashes),
                      body_unchanged_by_constraint=True,raw_observations_unchanged=True)
    (out/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    meta.update(club_refinement=refinement,window_blending=True,confidence_source=confidence_source,
                checkpoint_selection={**meta['checkpoint_selection'],'fresh_2d_inference':False},
                replay_source=str(source),video=str(avi),still=str(still),arrays=str(out/npz.name),
                frames_written=written,position_stills=stills)
    every=np.arange(len(raw)); both=np.flatnonzero(valid[:,17]&valid[:,18])
    meta['bone_length_3d']={f'{a}-{b}':workflow.length_stats(poses,both if a=='shaft' else every,a,b) for a,b in workflow.BONES}
    (out/npz.with_suffix('.json').name).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,4))
    for label,p in [('Previous',d['keypoints_3d']),('New inputs + blended windows',before),('Constrained',poses)]:
        ax.plot(np.linalg.norm(p[:,18]-p[:,17],axis=1),label=label)
    ax.set(xlabel='Video frame (zero based)',ylabel='Shaft–hosel distance (model metres)',
           title='Length consistency experiment; constrained length is inferred, not measured')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'length_comparison.png',dpi=150)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
