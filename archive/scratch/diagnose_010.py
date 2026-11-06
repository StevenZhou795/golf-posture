from pathlib import Path
import json,cv2,torch,numpy as np
from golf_workflows.official_yolox import OfficialGolferDetector,select_box
out=Path('runs/diagnose_010_20261104');out.mkdir(exist_ok=True)
torch.set_num_threads(8)
model=OfficialGolferDetector(score_threshold=.001)
cap=cv2.VideoCapture('pro_refs/010.mp4');count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
print('frames',count,'fps',fps,'size',cap.get(3),cap.get(4),flush=True)
rows=[]
for f in range(count):
 ok,frame=cap.read()
 if not ok:break
 r=model.predict(frame)
 rows.append(dict(frame_index=f,boxes=r['boxes'].tolist(),scores=r['scores'].tolist(),labels=r['labels'].tolist()))
 if f in (0,count//2,count-1):cv2.imwrite(str(out/f'frame_{f}.jpg'),frame)
 if f%20==0:print(f,'best',float(r['scores'].max()) if len(r['scores']) else None,flush=True)
cap.release()
d=dict(frames=len(rows),fps=fps,frames_with_box_at_025=sum(any(s>=.25 for s in r['scores']) for r in rows),details=rows)
(out/'detector.json').write_text(json.dumps(d,indent=2))
print('SUMMARY',len(rows),d['frames_with_box_at_025'],flush=True)
