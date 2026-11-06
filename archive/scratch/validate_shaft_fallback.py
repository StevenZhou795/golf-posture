from pathlib import Path
import json,shutil
from golf_workflows import standard_frames as reference
source=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
output=Path('runs/validation_shaft_fallback_20261103').resolve()
output.mkdir(exist_ok=False)
for position in ('P8','P10'):
 folder=next(source.glob(position+'-*'))
 shutil.copytree(folder,output/folder.name)
previous=reference.OUTPUT_DIR
try:
 reference.OUTPUT_DIR=output
 reference.rebuild_references()
finally:reference.OUTPUT_DIR=previous
for folder in output.iterdir():
 d=json.loads((folder/'joints.json').read_text())
 overlay=d['reference_club_overlay']
 assert overlay['applied'] and overlay['shaft_only_fallback']
 assert (folder/'reference_3d_club_overlay.jpg').exists()
 print(d['position'],overlay['source_view'],overlay['inferred_club_landmarks'])
