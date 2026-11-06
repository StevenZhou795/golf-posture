import numpy as np, json, sys
from pathlib import Path
from ultralytics import YOLO
from golf_workflows.club_measurements import shaft_angle_measurements
from golf_workflows.golf_pose_3d import MODEL_PATH
seg=YOLO(str(MODEL_PATH))
for name,run in [('w09','runs/workflows/videos/w09_autoroi_20261105/w09_3d'),('005','.tmp/club_005/005_3d'),('008','.tmp/club_008/008_3d')]:
    d=np.load(run+'.npz'); s=json.load(open(run+'.json'))
    video=Path(s['source'])
    P=d['keypoints_2d'].astype(float); v=d['keypoints_2d_valid']; F=d['keypoints_2d_filled'].astype(float)
    conf=d['club_landmark_confidence'][:,:2].min(1)
    hands=(F[:,13]+F[:,16])/2
    vec=P[:,18]-P[:,17]; L=np.linalg.norm(vec,axis=1); rel=v[:,17]&v[:,18]&(conf>=.5)
    Lp=np.median(L[rel]); ang=np.arctan2(vec[:,1],vec[:,0])
    m,c=shaft_angle_measurements(video,set(range(len(P))),hands,Lp,seg)
    print(name,c,'rel',rel.sum(),'unrel',int((~rel).sum()),'L',round(Lp))
    for src in ('streak','mask'):
        err=[];cov=0
        for f,lst in m.items():
            for a,sg,so in lst:
                if so==src and rel[f]: err.append(abs(np.degrees((a-ang[f]+np.pi)%(2*np.pi)-np.pi)))
                if so==src and not rel[f]: cov+=1
        err=np.array(err)
        if len(err): print(' ',src,'n',len(err),'median',round(np.median(err),1),'p80',round(np.percentile(err,80),1),'within15',round((err<15).mean(),2),'on unreliable',cov)
