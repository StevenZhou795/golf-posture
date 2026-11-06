from pathlib import Path
import json,numpy as np
new=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
old=Path('runs/workflows/references_20261102_173211_586764')
print('NEW',new)
for position in ('P1','P2','P7','P9','P10'):
 print(position)
 for label,root in [('previous',old),('joint',new)]:
  folder=next(root.glob(position+'-*'));d=json.loads((folder/'joints.json').read_text())
  with np.load(folder/'joints.npz') as z:
   p=z['face_on_2d']; q=z['face_on_3d']; hip2=np.linalg.norm(p[1]-p[4]); shoulder2=np.linalg.norm(p[11]-p[14]);hip3=np.linalg.norm(q[1]-q[4]); shoulder3=np.linalg.norm(q[11]-q[14]);
   detail=d['views']['face_on'].get('pose_backend_details') or {}; affine=detail.get('crop_affine')
   net_width=np.linalg.norm((p[1]-p[4])@np.array(affine)[:,:2].T) if affine else None
   print(label,'hip2px',round(float(hip2),2),'hip/shoulder2',round(float(hip2/shoulder2),3),'hip/shoulder3',round(float(hip3/shoulder3),3),'hip3',round(float(hip3),3),'network_hip_px',None if net_width is None else round(float(net_width),2),'detailkeys',list(detail) if position=='P1' else '')
