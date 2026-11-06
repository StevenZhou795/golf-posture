from pathlib import Path
import json,shutil,tempfile,numpy as np
from golf_workflows.standard_frames import build_reference_club_overlay,save_still,VIEWS
root=Path(json.loads(Path('runs/workflows/latest_references.json').read_text())['folder'])
folder=next(root.glob('P10-*'))
summary=json.loads((folder/'joints.json').read_text())
with np.load(folder/'joints.npz') as stored:
 arrays={key:stored[key] for key in stored.files}
before=arrays['reference_3d'].copy()
detections={view:dict(points=arrays[view+'_2d'],valid=arrays[view+'_2d_valid'],club_confidence=np.asarray(record['club_landmark_confidence']),club_pose_details=record.get('club_pose_details')) for view,record in summary['views'].items()}
fused=dict(reference=before,aligned={view:arrays[view+'_aligned'] for view in VIEWS})
overlay,metadata=build_reference_club_overlay(fused,detections)
assert metadata['applied'] and metadata['shaft_only_fallback'] and metadata['source_view']=='down_the_line'
np.testing.assert_array_equal(overlay[17:22],arrays['down_the_line_aligned'][17:22])
np.testing.assert_array_equal(overlay[:17],before[:17])
backup_root=Path('runs/workflow_snapshots');backup_root.mkdir(exist_ok=True)
backup=Path(tempfile.mkdtemp(prefix='p10_overlay_before_refresh_',dir=backup_root))
for name in ('joints.json','joints.npz','reference_3d_club_overlay.jpg'):
 if (folder/name).exists():shutil.copy2(folder/name,backup/name)
if (root/'run_summary.json').exists():shutil.copy2(root/'run_summary.json',backup/'run_summary.json')
arrays['reference_3d_club_overlay']=overlay
np.savez(folder/'joints.npz',**arrays)
summary['reference_club_overlay']=metadata
(folder/'joints.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
save_still(overlay,folder/'reference_3d_club_overlay.jpg','reference (club; inferred points)')
if (root/'run_summary.json').exists():
 run=json.loads((root/'run_summary.json').read_text())
 run['reference_club_overlays']=sum(json.loads(p.read_text()).get('reference_club_overlay',{}).get('applied',False) for p in root.glob('*/joints.json'))
 (root/'run_summary.json').write_text(json.dumps(run,indent=2),encoding='utf8')
with np.load(folder/'joints.npz') as stored:np.testing.assert_array_equal(stored['reference_3d'],before)
print('Updated:',folder/'reference_3d_club_overlay.jpg')
print('Backup:',backup)
print(json.dumps(metadata,indent=2))
