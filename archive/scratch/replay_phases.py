"""Replay saved 2D arrays through golf_pose_3d.process (phases, 17+2 lifting, extraction, rendering)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from golf_workflows import golf_pose_3d as g

run, video, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
view = sys.argv[4] if len(sys.argv) > 4 else 'down_the_line'
stem = video.stem
data = np.load(run / f'{stem}_3d.npz')
summary = json.loads((run / f'{stem}_3d.json').read_text(encoding='utf-8'))
points, valid = data['keypoints_2d'].astype(np.float64), data['keypoints_2d_valid']
boxes = []
for p, v in zip(points, valid):
    body = p[:17][v[:17]]
    boxes.append(np.concatenate([body.min(0), body.max(0)]) if len(body) >= 2 else np.full(4, np.nan))
wrist_y = np.array([np.nanmean([p[i, 1] for i in (13, 16) if v[i]]) if v[[13, 16]].any() else np.nan
                    for p, v in zip(points, valid)])
swing = g.Swing2D(frames=len(points), fps=float(data['fps']), size=tuple(int(x) for x in data['frame_size']),
    points=points, valid=valid, filled=g.fill_gaps(points, valid), boxes=np.stack(boxes), samples=[],
    wrist_y=wrist_y, club_confidence=data['club_landmark_confidence'],
    keypoint_confidence=data['keypoints_2d_confidence'] if 'keypoints_2d_confidence' in data else None)
g.detect_2d = lambda *args, **kwargs: swing
pose_model = SimpleNamespace(joint_pose=True, metadata=summary.get('pose_backend') or {})
out.mkdir(parents=True, exist_ok=True)
g.process(video, pose_model, None, g.load_lifter(g.CHECKPOINT), handedness=summary.get('handedness', 'right'),
          view=view, club_refinement=True, output_folder=out)
