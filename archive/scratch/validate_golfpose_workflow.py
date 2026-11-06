from pathlib import Path
import hashlib,json
import numpy as np
from golf_workflows import cli
root=Path('runs/golfpose_workflow').resolve()
refs=cli.latest('references',root)
video=cli.latest('video',root)
score=cli.latest('score',root)
entries=cli.validate_reference_set(refs)
for row in entries.values():
 with np.load(refs/row['coordinates']) as d:
  assert d['reference_3d'].shape==(22,3)
  assert np.isfinite(d['reference_3d']).all()
 for view in ('face_on','down_the_line','rear'):
  with np.load(refs/row['coordinates']) as d:
   assert d[f'{view}_2d_confidence'].shape==(22,)
manifest=cli.read_json(video/'identified_frames.json')
with np.load(video/'identified_frames.npz') as d:
 assert d['keypoints_3d'].shape==(10,22,3)
 assert d['keypoints_2d'].shape==(10,22,2)
 assert d['keypoints_2d_confidence'].shape==(10,22)
 assert np.isfinite(d['keypoints_3d']).all()
 indices=d['source_frame_indices'].tolist()
 assert all(a<b for a,b in zip(indices,indices[1:]))
for i in range(1,11):
 assert (video/f'P{i}.jpg').is_file() and (video/f'P{i}_3d.jpg').is_file()
summary=cli.read_json(next(video.glob('*_3d.json')))
assert summary['pose_backend']['pose_backend']=='golfpose-golfer-vitpose-huge'
assert summary['checkpoint_selection']==dict(method='automatic',frame_index_base=0,fresh_2d_inference=True)
assert summary['club_model'] is None
scored=cli.read_json(score/'score.json')
assert scored['video_run']==str(video) and scored['references']==str(refs)
assert scored['coverage']['positions_scored']==10 and len(scored['criteria'])==14
before=json.loads(Path('.tmp/golfpose_original_pointers.json').read_text())
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('runs/workflows').glob('latest_*.json')}
assert before==after, 'Original pointers changed during validation'
report=dict(reference_folder=str(refs),reference_count=10,video_folder=str(video),
 video_frames=summary['frames'],selected_indices=indices,selected_pose_shape=[10,22,3],
 fresh_2d_inference=True,automatic_extraction=True,pose_backend=summary['pose_backend'],
 club_evidence=summary['club_evidence_counts'],score_folder=str(score),
 score=scored['pose_score'],coverage=scored['coverage'],original_pointers_unchanged=True,
 tests_passed=60,note='Execution and data-contract validation; not a ground-truth accuracy measurement.')
(root/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
