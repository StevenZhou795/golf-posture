from pathlib import Path
import cv2,json,numpy as np
from golf_workflows.golfer_vitpose import GolferViTPose
from golf_workflows import standard_frames as reference
from golf_workflows import golf_pose_3d as video
from golf_workflows.cli import export_identified_frames
root=Path('runs/validation_golfer_vitpose_20261103').resolve()
root.mkdir(exist_ok=False,parents=True)
model=GolferViTPose()
lifter=video.load_lifter(video.CHECKPOINT_22)
old=reference.OUTPUT_DIR
reference.OUTPUT_DIR=root/'reference'
try:
 report=reference.process_sheet(next(reference.INPUT_DIR.glob('P1-*')),model,None,lifter,model)
finally: reference.OUTPUT_DIR=old
cap=cv2.VideoCapture('pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4')
count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS) or 30)
frames=[]
for index in np.linspace(0,count-1,10).astype(int):
 cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
 if not ok: raise RuntimeError('Could not read validation source')
 frames.append(frame)
cap.release()
clip=root/'ten_sampled_frames.mp4'
writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frames[0].shape[1],frames[0].shape[0]))
for frame in frames: writer.write(frame)
writer.release()
positions={f'p{i+1}':i for i in range(10)}
video.process(clip,model,None,lifter,reviewed_positions=positions,club_refinement=True,output_folder=root/'video',club_pose_model=model)
manifest=export_identified_frames(root/'video')
with np.load(root/'video'/'identified_frames.npz') as data:
 shape=list(data['keypoints_3d'].shape)
 assert shape==[10,22,3] and np.isfinite(data['keypoints_3d']).all()
with np.load(root/'video'/'ten_sampled_frames_3d.npz') as data:
 assert data['keypoints_2d'].shape==(10,22,2)
 assert data['keypoints_2d_confidence'].shape==(10,22)
result=dict(reference_finite=report['reference_3d'] is not None,video_selected_shape=shape,
 note='Plumbing validation using ten sampled frames and explicit test indices; not validation of automatic golf-phase selection.',models=model.metadata)
(root/'validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
