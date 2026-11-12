"""SwingTrace boxes -> explicit, uncalibrated shaft/head centre proxies.

Class ID 2 is labelled '3' in the released checkpoint. It is hands/grip,
not a fourth class. No box corner is promoted to a measured club endpoint.
"""
from pathlib import Path
from time import perf_counter

import numpy as np

from .runtime import ROOT

CHECKPOINT = ROOT/'models/swingtrace-yolo11m.pt'
CLASS_NAMES = {0: 'shaft', 1: 'head', 2: 'grip'}


def centre(box):
    box = np.asarray(box, dtype=float)
    return (box[:2] + box[2:]) / 2


def box_distance(point, box):
    box = np.asarray(box, dtype=float)
    return float(np.linalg.norm(np.maximum(np.maximum(box[:2]-point, point-box[2:]), 0)))


def associate(detections, anchor, person_box, previous_head=None, threshold=.25):
    """Associate detections to RTMW wrists, then require shaft/grip/head consistency.

    Missing detections remain missing. Geometry gates use golfer height so
    a bright line or another player's club cannot win solely on confidence.
    """
    selected = dict(shaft=None, head=None, grip=None)
    out = dict(selected=selected, pair_usable=False, reason='no_reliable_golfer_wrist',
               proxy_xy=[None, None], proxy_confidence=[0., 0.])
    if anchor is None or not np.isfinite(anchor).all() or not np.isfinite(person_box).all():
        return out
    anchor = np.asarray(anchor, dtype=float)
    height = max(1., float(person_box[3]-person_box[1]))
    pools = {name: [] for name in CLASS_NAMES.values()}
    for row in detections:
        b = np.asarray(row['box'], dtype=float)
        score = float(row['confidence'])
        if (row['class_id'] in CLASS_NAMES and b.shape == (4,) and np.isfinite(b).all()
                and np.all(b[2:] > b[:2]) and np.isfinite(score) and threshold <= score <= 1):
            pools[CLASS_NAMES[row['class_id']]].append(row)
    grips = [r for r in pools['grip'] if box_distance(anchor, r['box']) <= .20*height]
    if grips:
        selected['grip'] = min(grips, key=lambda r: np.linalg.norm(centre(r['box'])-anchor)/height-.1*r['confidence'])
    grip = centre(selected['grip']['box']) if selected['grip'] else anchor
    out['grip_anchor_source'] = 'swingtrace_grip_box' if selected['grip'] else 'rtmw_wrists'
    shafts = [r for r in pools['shaft'] if box_distance(grip, r['box']) <= .20*height]
    best = None
    for shaft in shafts:
        s = centre(shaft['box'])
        for head in pools['head']:
            h = centre(head['box'])
            vector = h-grip
            length = float(np.linalg.norm(vector))
            if not .08*height <= length <= 1.8*height:
                continue
            fraction = float(np.dot(s-grip, vector)/length**2)
            residual = float(np.linalg.norm(s-(grip+fraction*vector)))
            if not .05 <= fraction <= .95 or residual > max(6., .15*length):
                continue
            if box_distance(h, shaft['box']) > max(8., .20*length):
                continue
            score = min(float(shaft['confidence']), float(head['confidence'])) - residual/length
            if previous_head is not None:
                score -= .05*min(2., np.linalg.norm(h-previous_head)/height)
            if best is None or score > best[0]:
                best = (score, shaft, head, s, h)
    if best is not None:
        _, selected['shaft'], selected['head'], s, h = best
        # The two centres define a proxy direction, not GolfPose's physical markers.
        confidence = min(selected['shaft']['confidence'], selected['head']['confidence'],
                         selected['grip']['confidence'] if selected['grip'] else 1.)
        out.update(pair_usable=True, reason='associated_box_centres',
                   proxy_xy=[s.tolist(), h.tolist()], proxy_confidence=[confidence, confidence])
    elif shafts:
        selected['shaft'] = min(shafts, key=lambda r: box_distance(grip, r['box'])/height-.1*r['confidence'])
        out.update(reason='head_missing_or_geometry_rejected',
                   proxy_xy=[centre(selected['shaft']['box']).tolist(), None],
                   proxy_confidence=[selected['shaft']['confidence'], 0.])
    else:
        out['reason'] = 'no_shaft_associated_with_golfer'
    return out


class SwingTraceClub:
    confidence_threshold = .25

    def __init__(self, checkpoint=CHECKPOINT, device='cpu', confidence=.25):
        from ultralytics import YOLO
        import hashlib
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.model = YOLO(str(self.checkpoint), task='detect')
        if self.model.task != 'detect' or self.model.names != {0: '0', 1: '1', 2: '3'}:
            raise ValueError(f'Unexpected SwingTrace task/classes: {self.model.task}, {self.model.names}')
        self.confidence_threshold = float(confidence)
        self.metadata = dict(club_pose_backend='swingtrace-yolo11m',
            club_pose_checkpoint=str(self.checkpoint), sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            class_id_to_semantic=CLASS_NAMES, raw_class_names=self.model.names,
            input_size=640, precision='float32',
            landmark_semantics={'shaft': 'shaft box centre proxy', 'hosel': 'head box centre proxy'},
            caveat='Box centres are not trained GolfPose markers; MixSTE club geometry is approximate.',
            missing_policy='no box-corner recovery; no invented heel/toe; no ViTPose fallback')
        self.records = []
        self.previous_head = None
        self.set_device(device)

    def set_device(self, device):
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --club-device cpu')
        self.device = device
        self.model.to(device)
        self.model.model.float().eval()
        # Ultralytics caches its AutoBackend on the first predict call.
        self.model.predictor = None
        self.metadata['inference_device'] = device

    def reset(self):
        self.records = []
        self.previous_head = None

    def observe(self, frame, anchor, person_box, frame_index):
        import torch
        started = perf_counter()
        threads = torch.get_num_threads()
        try:
            result = self.model.predict(frame, device=self.device, imgsz=640,
                conf=self.confidence_threshold, half=False, verbose=False)[0]
        finally:
            if torch.get_num_threads() != threads:
                torch.set_num_threads(threads)
        boxes = result.boxes
        raw = [] if boxes is None else [dict(class_id=int(c), box=b.tolist(), confidence=float(s))
            for b, s, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]
        row = associate(raw, anchor, person_box, self.previous_head, self.confidence_threshold)
        if row['pair_usable']:
            self.previous_head = np.array(row['proxy_xy'][1])
        else:
            self.previous_head = None
        row.update(frame_index=int(frame_index), detections=raw, inference_seconds=perf_counter()-started)
        self.records.append(row)
        xy = np.full((2, 2), np.nan)
        for i, point in enumerate(row['proxy_xy']):
            if point is not None:
                xy[i] = point
        return xy, np.asarray(row['proxy_confidence'], dtype=np.float32), row

    def save(self, folder):
        from .report import write_json
        write_json(folder/'swingtrace_detections.json', dict(model=self.metadata, frames=self.records))

    def draw(self, frame, frame_index):
        from .overlay import draw_detections
        row = self.records[frame_index] if 0 <= frame_index < len(self.records) else None
        if row is not None and row['frame_index'] != frame_index:
            raise ValueError('SwingTrace render frame index does not match observations')
        return draw_detections(frame, row)


class SwingTraceSequence:
    """Use the existing joint-sequence protocol; bypass all ViTPose/crop calls."""
    joint_pose = True

    def __init__(self, body, club):
        self.body, self.club = body, club
        self.metadata = dict(body.metadata, club_crop_backend='swingtrace-full-frame',
                             club_observation_kind='box_centre_proxies')

    def detect_sequence(self, video, track_club_path=False):
        from golf_workflows import golf_pose_3d as shared
        rows, flags, scores, boxes, details, club_scores, wrist_y = [], [], [], [], [], [], []
        self.club.reset()
        for index, result in enumerate(self.body.predict(str(video), stream=True)):
            frame = result.orig_img
            size = (frame.shape[1], frame.shape[0])
            points = np.full((19, 2), np.nan)
            valid = np.zeros(19, bool)
            confidence = np.zeros(19, np.float32)
            person = shared.golfer_index(result)
            box = np.full(4, np.nan)
            anchor = None
            if person is not None:
                xy = result.keypoints.xy[person].cpu().numpy().astype(float)
                conf = result.keypoints.conf[person].cpu().numpy()
                points[:17], valid[:17] = shared.coco_to_golfpose(xy, conf)
                hands = [xy[i] for i in (9, 10) if conf[i] >= shared.KEYPOINT_CONFIDENCE and np.isfinite(xy[i]).all()]
                anchor = np.mean(hands, axis=0) if hands else None
                box = result.boxes.xyxy[person].cpu().numpy().astype(float)
                for name, coco in shared.DIRECT_FROM_COCO.items():
                    confidence[shared.JOINT[name]] = conf[coco]
            club_xy, club_conf, row = self.club.observe(frame, anchor, box, index)
            points[17:19] = club_xy
            valid[17:19] = np.isfinite(club_xy).all(axis=1) & (club_conf >= self.club.confidence_threshold)
            confidence[17:19] = club_conf
            rows.append(points)
            flags.append(valid)
            scores.append(confidence)
            boxes.append(box)
            club_scores.append(np.r_[club_conf, np.zeros(3, np.float32)])
            wrist_y.append(shared.wrist_height(result, person))
            details.append(dict(frame_index=index, body=getattr(result, 'detector_details', None), club=row))
            if index % 50 == 0:
                print(f'SwingTrace frame {index}', flush=True)
        if not rows:
            raise ValueError('No decoded frames')
        points, valid = np.stack(rows), np.stack(flags)
        if not valid[:, shared.JOINT['root']].any():
            raise ValueError('No golfer with both hips found')
        return shared.Swing2D(frames=len(rows), fps=float(self.body.predictor.dataset.fps), size=size,
            points=points, valid=valid, filled=shared.fill_gaps(points, valid), boxes=np.stack(boxes),
            samples=[], wrist_y=np.asarray(wrist_y), club_confidence=np.stack(club_scores),
            keypoint_confidence=np.stack(scores), detector_details=details, predicted_shaft_crops=[])
