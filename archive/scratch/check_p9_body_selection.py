import json,numpy as np
from pathlib import Path
from golf_workflows.standard_frames import fuse_views,VIEWS
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
folder=next(root.glob('P9-*'))
d=json.loads((folder/'joints.json').read_text())
with np.load(folder/'joints.npz') as z:
 poses={v:z[v+'_3d'] for v in VIEWS}
 detections={v:dict(valid=z[v+'_2d_valid'], keypoint_confidence=z[v+'_2d_confidence'], pose_backend_details=d['views'][v]['pose_backend_details']) for v in VIEWS}
 result=fuse_views(poses,detections)
 assert result['body_selection']['source_view']=='face_on'
 np.testing.assert_array_equal(result['reference'],result['aligned']['face_on'])
 print(json.dumps(result['body_selection'],indent=2))
