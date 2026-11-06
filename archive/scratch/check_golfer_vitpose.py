from golf_workflows.golfer_vitpose import GolferViTPose
from golf_workflows.standard_frames import crop_panels, INPUT_DIR
import cv2, json
m=GolferViTPose()
print('output channels',m.model.head.final_layer.out_channels,flush=True)
print('\n'.join(s for s in str(m.model.source_config).splitlines() if 'out_channels=' in s or 'golfswing_golfer' in s),flush=True)
p=crop_panels(cv2.imread(str(next(INPUT_DIR.glob('P1-*')))))[0]
d=m.predict_frame(p)
print('reference valid',d['valid'].sum(),'club valid',d['valid'][17:].tolist(),flush=True)
print('confidence',d['keypoint_confidence'].tolist(),flush=True)
cap=cv2.VideoCapture('pro_refs/b85ed0b4db566e63bce190824c2f2196.mp4')
ok,frame=cap.read();cap.release()
d=m.predict_frame(frame)
print('video valid',d['valid'].sum(),'club valid',d['valid'][17:].tolist(),flush=True)
