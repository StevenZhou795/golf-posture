import json, numpy as np
from pathlib import Path
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
for folder in sorted(root.iterdir()):
 if not (folder/'joints.json').exists():continue
 d=json.loads((folder/'joints.json').read_text())
 print(d['position'], 'selected', d['reference_club_overlay'].get('source_view'))
 for name,v in d['views'].items():
  p=v['joints_2d']; c=np.asarray(v['club_landmark_confidence']); a=p.get('shaft');b=p.get('hosel')
  shaft=round(float(np.linalg.norm(np.array(a)-b)),1) if a is not None and b is not None else None
  score=round(float(c.min()+.25*np.clip(c,0,1).mean()),3)
  print(' ',name,'score',score,'min',round(float(c.min()),3),'shaft_px',shaft,'all5',all(p.get(k) is not None for k in ['shaft','hosel','heel','toe_down','toe_up']))
