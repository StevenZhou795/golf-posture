from pathlib import Path
import json,cv2,torch,numpy as np
from golf_workflows.official_yolox import OfficialClubDetector,OfficialGolferDetector,select_box
from golf_workflows.standard_frames import crop_panels,INPUT_DIR,sheet_order,VIEWS
torch.set_num_threads(8)
out=Path('runs/validation_detector_routing_20261103');out.mkdir(exist_ok=True)
report={}
for kind in (OfficialClubDetector,OfficialGolferDetector):
 model=kind();refs=[];video=[]
 for path in sorted(INPUT_DIR.glob('*.jpg'),key=sheet_order):
  for view,frame in zip(VIEWS,crop_panels(cv2.imread(str(path)))):
   r=model.predict(frame);box,score=select_box(r)
   refs.append(dict(position=path.stem,view=view,box=None if box is None else box.tolist(),score=score))
 cap=cv2.VideoCapture('pro_refs/w07.mp4');previous=None
 while True:
  ok,frame=cap.read()
  if not ok:break
  r=model.predict(frame);box,score=select_box(r,previous=previous)
  if box is not None:previous=box
  video.append(dict(frame_index=len(video),box=None if box is None else box.tolist(),score=score))
 cap.release()
 report[kind.__name__]=dict(reference_boxes=sum(r['box'] is not None for r in refs),reference_views=len(refs),
  video_boxes=sum(r['box'] is not None for r in video),video_frames=len(video),reference_details=refs,video_details=video)
 print(kind.__name__,{k:v for k,v in report[kind.__name__].items() if not k.endswith('details')},flush=True)
(out/'detector_audit.json').write_text(json.dumps(report,indent=2))
