from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,torch
from golf_workflows import golf_pose_3d as workflow
from golf_workflows.cli import export_identified_frames
source=Path(json.loads(Path('runs/workflows/latest_video.json').read_text())['folder'])
npz=next(source.glob('*_3d.npz')); saved=json.loads(npz.with_suffix('.json').read_text())
with np.load(npz) as z:
 points=z['keypoints_2d'].copy(); valid=z['keypoints_2d_valid'].copy(); confidence=z['keypoints_2d_confidence'].copy();fps=float(z['fps']);size=tuple(z['frame_size'].tolist())
boxes=[];wrists=[]
for p,v in zip(points,valid):
 body=p[:17][v[:17]];boxes.append(np.r_[body.min(0),body.max(0)])
 ids=[i for i in (13,16) if v[i]];wrists.append(p[ids,1].mean() if ids else np.nan)
swing=workflow.Swing2D(frames=len(points),fps=fps,size=size,points=points,valid=valid,filled=workflow.fill_gaps(points,valid),boxes=np.array(boxes),samples=[],wrist_y=np.array(wrists),club_confidence=confidence[:,17:],keypoint_confidence=confidence)
model=SimpleNamespace(joint_pose=True,metadata=saved['pose_backend'],detect_sequence=lambda *args,**kwargs:swing)
out=Path('runs/validation_video_reference_parity_20261103').resolve();out.mkdir(exist_ok=False)
torch.set_num_threads(8)
lifter=workflow.load_lifter(workflow.CHECKPOINT_22)
workflow.process(Path(saved['source']),model,None,lifter,club_pose_model=model,club_refinement=True,output_folder=out)
summary_path=next(out.glob('*_3d.json'));d=json.loads(summary_path.read_text());d['checkpoint_selection']['fresh_2d_inference']=False;d['validation_reused_2d_from']=str(npz);summary_path.write_text(json.dumps(d,indent=2))
manifest=export_identified_frames(out)
with np.load(out/'identified_frames.npz') as z:
 assert z['keypoints_3d'].shape==(10,22,3) and np.isfinite(z['keypoints_3d']).all()
 assert z['keypoints_2d_confidence'].shape==(10,22)
print('AUTOMATIC POSITIONS',d['positions'],flush=True)
print('EVIDENCE',d['club_evidence_counts'],flush=True)
print('SOURCE',d['phase_detection']['club_track_source'],flush=True)
