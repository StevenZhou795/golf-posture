from pathlib import Path
import shutil,json,numpy as np
from golf_workflows import standard_frames as reference
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
source=next(root.glob('P9-*'))
out=Path('runs/validation_body_selection_20261103').resolve()
out.mkdir(exist_ok=False)
shutil.copytree(source,out/source.name)
previous=reference.OUTPUT_DIR
try:
 reference.OUTPUT_DIR=out
 reference.rebuild_references()
finally: reference.OUTPUT_DIR=previous
with np.load(out/source.name/'joints.npz') as z:
 np.testing.assert_array_equal(z['reference_3d'],z['face_on_aligned'])
 np.testing.assert_array_equal(z['reference_3d_club_overlay'][:17],z['reference_3d'][:17])
 with np.load(source/'joints.npz') as old:
  for v in reference.VIEWS: np.testing.assert_array_equal(z[v+'_3d'],old[v+'_3d'])
print('Cached P9 rebuild preserves complete face-on geometry and original camera poses.')
