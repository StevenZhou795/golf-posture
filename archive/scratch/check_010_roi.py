from pathlib import Path
import json,cv2,numpy as np
from golf_workflows.golfer_vitpose import GolferViTPose
from golf_workflows.standard_frames import draw_labeled
root=Path('runs/diagnose_010_20261104')
model=GolferViTPose(cpu_threads=8)
box=np.array([140.,220.,500.,900.]);model.set_initial_golfer_box(box)
cap=cv2.VideoCapture('pro_refs/010.mp4');report=[]
for index in (0,126,200):
 cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
 if not ok:raise RuntimeError('Decode failed')
 r=model.predict_frame(frame,previous_box=box,allow_panel_fallback=False,previous_box_source='manual')
 row=dict(frame_index=index,crop_source=r['detector_details']['crop_source'],root_confidence=float(r['keypoint_confidence'][0]),root_valid=bool(r['valid'][0]),observed_joints=int(r['valid'].sum()),root_xy=r['points'][0].tolist())
 report.append(row);print(row,flush=True)
 cv2.imwrite(str(root/f'roi_frame_{index}_labeled.jpg'),draw_labeled(frame,r,f'frame {index}'))
cap.release();(root/'manual_roi_samples.json').write_text(json.dumps(report,indent=2))
