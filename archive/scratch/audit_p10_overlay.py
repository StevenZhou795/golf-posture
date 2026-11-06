from pathlib import Path
import json,numpy as np
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
for folder in root.glob('P10-*'):
 d=json.loads((folder/'joints.json').read_text()); print('FOLDER',folder); print(json.dumps(d['reference_club_overlay'],indent=2))
 with np.load(folder/'joints.npz') as z:
  for view,v in d['views'].items():
   print(view,'valid',z[view+'_2d_valid'][17:22].tolist(),'confidence',v['club_landmark_confidence'],'shaft',z[view+'_2d'][17].tolist())
  if 'reference_3d_club_overlay' in z:
   print('overlay club',z['reference_3d_club_overlay'][17:22].tolist())
   print('DTL club',z['down_the_line_aligned'][17:22].tolist())
