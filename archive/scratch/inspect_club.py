import numpy as np, json
d = np.load(r'runs/golfpose_workflow/videos/010_roi_20261104/010_3d.npz', allow_pickle=True)
for k in d.files: print(k, d[k].shape, d[k].dtype)
p = d['keypoints_2d']; v = d['keypoints_2d_valid']; c = d['keypoints_2d_confidence']
names = list(d['joints'])
print(names)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('median conf per joint', np.nanmedian(c,axis=0))
print('valid frac per joint', v.mean(0))
L = np.linalg.norm(p[:,18]-p[:,17],axis=1)
fa = np.linalg.norm(p[:,13]-p[:,12],axis=1)
print('shaft-hosel len pct', np.nanpercentile(L,[5,25,50,75,95]))
print('forearm len pct', np.nanpercentile(fa,[5,25,50,75,95]))
# head cluster spread
for j in (19,20,21):
    print(names[j], 'dist to hosel pct', np.nanpercentile(np.linalg.norm(p[:,j]-p[:,18],axis=1),[5,50,95]))
# distance from shaft pt to wrists
w = (p[:,13]+p[:,16])/2
print('shaft-to-wristmid pct', np.nanpercentile(np.linalg.norm(p[:,17]-w,axis=1),[5,50,95]))
for f in range(0,252,6):
    print(f, 'L=%.0f'%L[f], 'conf', c[f,17:22], 'shaft-wrist %.0f'%np.linalg.norm(p[f,17]-w[f]))
