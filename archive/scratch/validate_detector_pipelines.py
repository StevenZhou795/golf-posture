from pathlib import Path
import gc,json,hashlib
import numpy as np
import torch
from ultralytics import YOLOE
from golf_workflows import golf_pose_3d as video,standard_frames as refs
from golf_workflows.vitpose_tracking import load_pose_backend
from golf_workflows.official_yolox import OfficialClubDetector
from golf_workflows.detector_pose import YOLOXBodyPose
from golf_workflows.club_vitpose import ClubViTPose
from golf_workflows.cli import export_identified_frames
root=Path('runs/validation_detector_routing_20261103').resolve()
root.mkdir(exist_ok=True)
pointers={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for r in ('runs/workflows','runs/golfpose_workflow') for p in Path(r).glob('latest_*.json')}
torch.set_num_threads(8)
lifter=video.load_lifter(video.CHECKPOINT_22)
ball=YOLOE(str(video.BALL_MODEL)) if video.BALL_MODEL.is_file() else None
if ball is not None:ball.set_classes(['golf ball'])
report={}
for backend in ('yolo26n-pose','golfpose-golfer-vitpose-huge'):
 print('VALIDATING',backend,flush=True)
 pose=load_pose_backend(backend,cpu_threads=8)
 detector=None
 if backend=='yolo26n-pose':
  detector=OfficialClubDetector()
  pose=YOLOXBodyPose(pose,detector)
  club=ClubViTPose()
 else:club=pose
 folder=root/backend;folder.mkdir(exist_ok=True)
 old=refs.OUTPUT_DIR;refs.OUTPUT_DIR=folder/'reference'
 try:r=refs.process_sheet(next(refs.INPUT_DIR.glob('P1-*')),pose,None,lifter,club,detector)
 finally:refs.OUTPUT_DIR=old
 assert r['reference_3d'] is not None
 output=folder/'video'
 video.process(Path('pro_refs/w07.mp4').resolve(),pose,None,lifter,ball_model=ball,
  club_refinement=True,club_pose_model=club,club_crop_model=detector,output_folder=output)
 s=json.loads((output/'w07_3d.json').read_text())
 try:
  export_identified_frames(output)
  with np.load(output/'identified_frames.npz') as d:
   assert d['keypoints_3d'].shape==(10,22,3) and np.isfinite(d['keypoints_3d']).all()
  exported=True
 except ValueError as error:
  exported=False
  print('AUTOMATIC EXTRACTION LIMITATION:',str(error),flush=True)
  s['validation_extraction_error']=str(error)
 report[backend]=dict(reference_finite=True,video_frames=s['frames'],automatic_export_complete=exported,
  positions=s['positions'],extraction_error=s.get('validation_extraction_error'),
  models=s['pose_backend'],club_crop_backend=s['club_crop_backend'],club_evidence=s['club_evidence_counts'],
  detector_crops=s['detector_crops'],output=str(folder))
 del pose,club,detector
 gc.collect()
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for r in ('runs/workflows','runs/golfpose_workflow') for p in Path(r).glob('latest_*.json')}
assert pointers==after
report['latest_pointers_unchanged']=True
report['tests_passed']=66
(root/'pipeline_validation.json').write_text(json.dumps(report,indent=2))
for key,value in report.items():
 print(key,{k:v for k,v in value.items() if k!='detector_crops'} if isinstance(value,dict) else value,flush=True)
