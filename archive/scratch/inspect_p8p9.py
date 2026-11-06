from pathlib import Path
import json,numpy as np,cv2
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
print('RUN',root)
for folder in root.iterdir():
 if not (folder/'joints.json').exists():continue
 d=json.loads((folder/'joints.json').read_text())
 if d['position'] not in ('p8','p9'):continue
 print('POSITION',d['position']); print('overlay',json.dumps(d.get('reference_club_overlay')))
 print('kept',d.get('reference_kept'),'side_axis',d.get('side_axis'))
 with np.load(folder/'joints.npz') as z:
  for view,v in d['views'].items():
   conf=np.asarray(v['pose_backend_details']['confidence']); flags=z[view+'_2d_valid']; raw=z[view+'_2d']; im=cv2.imread(str(folder/(view+'_labeled.jpg')))
   print(view,'image_wh',[im.shape[1],im.shape[0]],'body_mean',float(conf[:17].mean()),'body_min',float(conf[:17].min()),'all_mean',float(conf.mean()),'club_min',float(conf[17:].min()),'club_score',float(conf[17:].min()+.25*np.clip(conf[17:],0,1).mean()),'valid_count',int(flags.sum()),'alignment',v.get('alignment'))
   for j in range(17,22):
    print(d['joints'][j],raw[j].tolist(),'conf',float(conf[j]),'valid',bool(flags[j]))
