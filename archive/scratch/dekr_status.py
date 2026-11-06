import json
from pathlib import Path
root=Path('runs/workflow_snapshots/vitpose_reference_20261102')
import hashlib
manifest={'saved_on':'2026-11-02','reference_pointer':json.loads((root/'latest_references.json').read_text()),'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.txt') and '__pycache__' not in p.parts}}
(root/'snapshot.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
for p in sorted(Path('runs/workflows/references_dekr_trial_20261102').glob('*/joints.json')):
 d=json.loads(p.read_text());print(p.parent.name,[(v,round(max(r['pose_backend_details']['confidence'][:17]),4)) for v,r in d['views'].items()])
