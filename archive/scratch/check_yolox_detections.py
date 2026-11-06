from pathlib import Path
import cv2,torch
from golf_workflows.official_yolox import OfficialClubDetector,OfficialGolferDetector
from golf_workflows.standard_frames import crop_panels,INPUT_DIR
torch.set_num_threads(8)
frame=crop_panels(cv2.imread(str(next(INPUT_DIR.glob('P1-*')))))[0]
cap=cv2.VideoCapture('pro_refs/w07.mp4');ok,video=cap.read();cap.release()
for kind in (OfficialClubDetector,OfficialGolferDetector):
 model=kind()
 for name,img in [('P1',frame),('w07_frame0',video)]:
  r=model.predict(img)
  print(kind.__name__,name,[(b.round(1).tolist(),round(float(s),4),int(l)) for b,s,l in zip(r['boxes'],r['scores'],r['labels'])][:8],flush=True)
