from pathlib import Path
from collections import Counter
import json,numpy as np
root=Path('runs/diagnose_010_20261104')
folder=Path('runs/golfpose_workflow/videos/010_roi_20261104').resolve()
s=json.loads((folder/'010_3d.json').read_text())
with np.load(folder/'010_3d.npz') as d:
 valid=d['keypoints_2d_valid'];confidence=d['keypoints_2d_confidence']
 assert d['keypoints_2d'].shape==(252,22,2)
 assert valid[:,0].any()
 count=int(valid[:,0].sum())
 root_scores=confidence[:,0]
 assert s['pose_backend']['initial_golfer_box_xyxy']==[140.,220.,500.,900.]
 status=(folder/'identified_frames.npz').is_file()
 indices=None
 if status:
  with np.load(folder/'identified_frames.npz') as selected:
   assert selected['keypoints_3d'].shape==(10,22,3) and np.isfinite(selected['keypoints_3d']).all()
   indices=selected['source_frame_indices'].tolist()
   assert all(a<b for a,b in zip(indices,indices[1:]))
report=dict(source=s['source'],video_frames=s['frames'],initial_golfer_box=[140,220,500,900],
 roi_sources=dict(Counter(row['crop_source'] for row in s['detector_crops'])),
 observed_root_frames=count,root_confidence_min=float(np.nanmin(root_scores)),root_confidence_max=float(np.nanmax(root_scores)),
 root_error_resolved=True,automatic_ten_frame_export=status,positions=s['positions'],selected_indices=indices,
 output=str(folder),tests_passed=72,note='Execution validation; joint confidence is not ground-truth pose accuracy.')
(root/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
