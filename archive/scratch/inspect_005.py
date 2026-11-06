import sys; sys.path.insert(0, '.')
import numpy as np, cv2
d = np.load('runs/golfpose_workflow/videos/005/005_3d.npz')
p, v = d['keypoints_2d_filled'], d['keypoints_2d_valid']
fps = float(d['fps'])
w = (p[:,13]+p[:,16])/2
torso = np.median(np.linalg.norm(p[:,8]-p[:,0],axis=1))
from golf_workflows.swing_phases import smooth_series
h = np.column_stack([smooth_series((w-p[:,0])[:,i]/torso,fps) for i in range(2)])
height = -h[:,1]
speed = np.linalg.norm(np.gradient(h,axis=0),axis=1)*fps
headd = np.linalg.norm(p[:,13]-p[:,10],axis=1)/torso
print('fps',fps,'frames',len(p),'p95 speed %.2f thr %.2f'%(np.percentile(speed,95), max(.08,.18*np.percentile(speed,95))))
for f in range(170, len(p), 8):
    print(f, 'height %.2f speed %.2f leadwrist-head %.2f valid_body %d' % (height[f], speed[f], headd[f], v[f,:17].sum()))
cap = cv2.VideoCapture('pro_refs/005.mp4')
want = [254, 270, 290, 310, 330, 360, 400, 450]
i = 0; tiles=[]
while True:
    ok, fr = cap.read()
    if not ok: break
    if i in want:
        t = cv2.resize(fr, (fr.shape[1]//3, fr.shape[0]//3)); cv2.putText(t, str(i), (10,30), 0, 1, (0,0,255), 2); tiles.append(t)
    i += 1
cv2.imwrite('.tmp/005_late.jpg', np.hstack(tiles))
