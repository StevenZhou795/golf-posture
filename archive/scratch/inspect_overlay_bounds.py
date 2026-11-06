import json,cv2,numpy as np
from pathlib import Path
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
for pos,view in [('p8','face_on'),('p10','down_the_line')]:
 for folder in root.iterdir():
  if not (folder/'joints.json').exists():continue
  d=json.loads((folder/'joints.json').read_text())
  if d['position']!=pos:continue
  v=d['views'][view]; print(pos,view)
  details=v['pose_backend_details']
  for i,k in enumerate(['shaft','hosel','heel','toe_down','toe_up'],17):
   print(k,details['landmark_xy'][i],details['confidence'][i],details['in_frame'][i])
  print('image',folder/f'{view}_labeled.jpg')
