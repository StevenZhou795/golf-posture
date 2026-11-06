"""Compare 22-joint ViTPose club points with GolfPose's 2-class YOLOX + club ViTPose on 010.mp4."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from golf_workflows.official_yolox import OfficialClubDetector, OfficialGolferDetector, select_box
from golf_workflows.club_vitpose import ClubViTPose

torch.set_num_threads(10)
RUN = ROOT / 'runs/golfpose_workflow/videos/010_roi_20261104'
OUT = ROOT / '.tmp/club_compare_010'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = [int(x) for x in sys.argv[1:]] or [30, 60, 72, 90, 103, 115, 126, 132, 139, 145, 150, 160, 167, 175, 180, 196, 210, 230]

saved = np.load(RUN / '010_3d.npz')
p22, c22 = saved['keypoints_2d'], saved['keypoints_2d_confidence']
club_det = OfficialClubDetector()
golfer_det = OfficialGolferDetector()
club_pose = ClubViTPose()
capture = cv2.VideoCapture(str(ROOT / 'pro_refs/010.mp4'))
rows, tiles = [], []
previous = np.array([140., 220., 500., 900.])
for f in FRAMES:
    capture.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = capture.read()
    assert ok, f
    golfer, _ = golfer_det.golfer_box(frame, previous)
    golfer = previous if golfer is None else golfer
    result = club_det.predict(frame)
    anchor = (p22[f, 13] + p22[f, 16]) / 2
    box, details = club_det.club_box(frame, result=result, person_box=golfer, anchor=anchor)
    if box is not None:
        xy, conf, info = club_pose.predict(frame, box)
        source = 'club-box'
    else:
        xy, conf, info = club_pose.predict_coarse_to_fine(frame, golfer)
        source = info.get('crop_fallback')
    a, b = p22[f, 17:22], xy
    rows.append(dict(frame=f, source=source, club_box=None if box is None else box.round(1).tolist(),
        club_score=details['selected_club_score'], club_candidates=details['club_candidates'],
        conf22=c22[f, 17:22].round(2).tolist(), conf_club=conf.round(2).tolist(),
        len22=float(np.linalg.norm(a[1] - a[0])), len_club=float(np.linalg.norm(b[1] - b[0])),
        hosel_gap=float(np.linalg.norm(a[1] - b[1])), shaft_gap=float(np.linalg.norm(a[0] - b[0])),
        xy22=a.round(1).tolist(), xy_club=b.round(1).tolist()))
    print(rows[-1], flush=True)
    view = frame.copy()
    if box is not None:
        cv2.rectangle(view, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 255, 0), 1)
    for pts, color in ((a, (0, 0, 255)), (b, (255, 255, 0))):
        q = pts.astype(int)
        cv2.line(view, tuple(q[0]), tuple(q[1]), color, 2)
        for i in range(1, 4):
            cv2.line(view, tuple(q[i]), tuple(q[i + 1]), color, 1)
        for i, point in enumerate(q):
            cv2.circle(view, tuple(point), 4 if i < 2 else 2, color, -1)
    cv2.putText(view, f'{f} red=22kp cyan=club-ViTPose ({source})', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / f'frame_{f:03d}.jpg'), view[150:1000])
    tiles.append(cv2.resize(view[150:1000], (296, 425)))
capture.release()
(OUT / 'comparison.json').write_text(json.dumps(rows, indent=2))
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
cv2.imwrite(str(OUT / 'comparison.jpg'), grid)
