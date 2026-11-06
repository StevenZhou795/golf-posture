import numpy as np, json
old = np.load('runs/golfpose_workflow/videos/010_roi_20261104/010_3d.npz')
new = np.load('runs/golfpose_workflow/videos/010_17plus2_20261104/010_3d.npz')
s = json.load(open('runs/golfpose_workflow/videos/010_17plus2_20261104/010_3d.json'))
w = json.load(open('runs/golfpose_workflow/videos/010_17plus2_20261104/workflow.json'))
print('checkpoint', s['checkpoint'], 'joint_count', s['joint_count'], 'dropped', s['dropped_2d_landmarks'])
print('evidence', s['club_evidence_counts'], 'refinement', {k: s['club_refinement'][k] for k in ('applied','reliable_pairs','marker_length_m')})
print('workflow club_landmarks', w['club_landmarks'], [f['club_evidence'] for f in w['frames']])
for k in ('keypoints_2d','keypoints_2d_confidence','keypoints_3d','club_landmark_confidence'):
    print(k, old[k].shape, '->', new[k].shape)
print('2D identical on shared 19 joints:', np.allclose(old['keypoints_2d'][:, :19], new['keypoints_2d'], equal_nan=True))
lo, hi = s['lift_window']
def club_len(a): return np.linalg.norm(a[:,18]-a[:,17],axis=1)
for name, d in (('17+5', old), ('17+2', new)):
    pre = d['keypoints_3d_before_club_refinement'][lo:hi+1]
    L = club_len(pre)
    body = pre[:, :17]
    print(name, 'pre-refinement 3D shaft-hosel length: median %.3f  CV %.3f  p5-p95 %.3f-%.3f' % (np.median(L), L.std()/L.mean(), *np.percentile(L,[5,95])))
    jerk = np.linalg.norm(np.diff(body, n=2, axis=0), axis=2).mean()
    print(name, 'body mean accel (smoothness) %.5f' % jerk)
print('body 3D mean diff 17+5 vs 17+2 (m): %.4f' % np.linalg.norm(old['keypoints_3d'][lo:hi+1,:17]-new['keypoints_3d'][lo:hi+1,:17],axis=2).mean())
