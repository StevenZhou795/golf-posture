"""Joint GolfPose ViTPose-H inference: 17 body and five club landmarks.

The one-class GolfPose YOLOX detector provides the golfer-with-club crop.
Decode into original image coordinates. Video gaps retain the last detected ROI.
Before the first detection, an initial ROI (manual --golfer-box, or by default
an automatic YOLO26n golfer box) is used; without one, joints stay unobserved.
"""
from pathlib import Path
from time import perf_counter
import json

import cv2
import numpy as np
import torch

from golf_workflows.vitpose_tracking import ROOT, crop_person, decode_heatmaps, load_checkpoint

CHECKPOINT = ROOT / 'golfpose_checkpoints' / 'golfpose_golfer_ViTPose_huge.pth'
BACKEND = 'golfpose-golfer-vitpose-huge'
FLIP = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13, 17, 18, 19, 20, 21]
FUSION_VITPOSE_BELOW = 0.3
FUSION_YOLO_ABOVE = 0.5


def validate_golfer_box(values):
    box = np.asarray(values, dtype=float)
    if (box.shape != (4,) or not np.isfinite(box).all() or (box[:2] < 0).any()
            or not (box[2] > box[0] and box[3] > box[1])):
        raise ValueError('--golfer-box requires finite pixel coordinates X1 Y1 X2 Y2, with 0 <= X1 < X2 and 0 <= Y1 < Y2')
    return box.copy()


class GolferViTPose:
    joint_pose = True
    confidence_threshold = .10

    def __init__(self, checkpoint=CHECKPOINT, cpu_threads=8):
        if cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        torch.set_num_threads(cpu_threads)
        from golf_workflows.official_yolox import OfficialGolferDetector
        self.detector = OfficialGolferDetector()
        self.model = load_checkpoint(Path(checkpoint), keypoints=22)
        self.initial_golfer_box = None
        self.diagnostics_dir = None
        self.body_fallback_model = None
        from golf_workflows.golf_pose_3d import joint_names
        self.metadata = dict(pose_backend=BACKEND, pose_checkpoint=str(Path(checkpoint).relative_to(ROOT)),
            club_pose_backend=BACKEND, club_crop_backend='official-yolox-1cls',
            joints=joint_names(22), joint_count=22, input_size_hw=[256, 192],
            decoder='UDPHeatmap + DARK refinement', flip_test=True,
            strict_parameter_load=True, cpu_threads=cpu_threads,
            confidence_threshold=self.confidence_threshold,
            crop_source='one-class YOLOX golfer box; UDP bbox expansion 1.25',
            person_detector=self.detector.metadata,
            detector_miss_policy='video: retain last detected ROI; initial miss unobserved; reference: full-panel fallback',
            out_of_frame_policy='raw coordinates retained, marked invalid')

    def set_initial_golfer_box(self, values, source='manual', details=None):
        if values is None:
            self.initial_golfer_box = None
            self.metadata.pop('initial_golfer_box_xyxy', None)
            self.metadata['initial_roi_source'] = None
            self.metadata['initial_roi_details'] = details
            return
        self.initial_golfer_box = validate_golfer_box(values)
        self.metadata['initial_golfer_box_xyxy'] = self.initial_golfer_box.tolist()
        self.metadata['initial_roi_source'] = source
        self.metadata['initial_roi_details'] = details

    @torch.inference_mode()
    def infer_crop(self, frame, crop_box):
        """22 landmarks (original pixels) and confidences from one flip-tested crop."""
        image, center, scale, matrix = crop_person(frame, crop_box)
        output = self.model(torch.cat([image, image.flip(-1)]))
        heatmap = (output[0] + output[1, FLIP].flip(-1)) / 2
        xy, confidence = decode_heatmaps(heatmap.cpu().numpy(), center, scale)
        return xy, confidence, matrix

    def refine_club(self, frame, detected, predictor, index):
        """Second pass on a crop around the predicted shaft when the club is weak.

        Keeps the five club landmarks from the second pass only when the weaker
        of shaft/hosel confidence improves; body joints stay from the first pass.
        """
        from golf_workflows.swing_phases import RELIABLE_CLUB_CONFIDENCE
        first = float(np.min(detected['keypoint_confidence'][17:19]))
        if first >= RELIABLE_CLUB_CONFIDENCE or not detected.get('found', False):
            return None
        hands = detected['points'][[13, 16]][detected['valid'][[13, 16]]]
        prediction = predictor.predict_box(index, hands.mean(axis=0) if len(hands) else None,
                                           (frame.shape[1], frame.shape[0]))
        if prediction is None:
            return None
        box, details = prediction
        xy, confidence, _matrix = self.infer_crop(frame, box)
        second = float(np.min(confidence[17:19]))
        details.update(frame_index=index, min_confidence_before=round(first, 3), min_confidence=round(second, 3),
                       kept=bool(second > first))
        if second > first:
            height, width = frame.shape[:2]
            club = slice(17, 22)
            detected['points'][club] = xy[club]
            detected['keypoint_confidence'] = detected['keypoint_confidence'].copy()
            detected['keypoint_confidence'][club] = confidence[club]
            in_frame = (np.isfinite(xy[club]).all(axis=1) & (xy[club, 0] >= 0) & (xy[club, 0] < width)
                        & (xy[club, 1] >= 0) & (xy[club, 1] < height))
            detected['valid'][club] = in_frame & (confidence[club] >= self.confidence_threshold)
            detected['club_confidence'] = confidence[club]
            detected['grip'] = xy[17] if detected['valid'][17] else None
            detected['head'] = xy[18] if detected['valid'][18] else None
        return details

    @staticmethod
    def body_usable(detected):
        """At least 12 of 17 body joints, both shoulders and a wrist."""
        valid = detected['valid']
        return bool(valid[:17].sum() >= 12 and valid[11] and valid[14] and (valid[13] or valid[16]))

    def recover_body(self, frame, detected, last_body_box):
        """Retry around the last complete skeleton, then the golfer ROI, when the crop lost the body.

        The golfer box can slide onto spectators or background during a fast
        swing; keeps the retry only if it yields a usable skeleton with more joints.
        """
        if self.body_usable(detected):
            return None
        candidates = []
        if last_body_box is not None:
            center, half = (last_body_box[:2] + last_body_box[2:]) / 2, (last_body_box[2:] - last_body_box[:2]) * .8
            candidates.append(('recovered-last-skeleton', np.r_[center - half, center + half]))
        if getattr(self, 'initial_golfer_box', None) is not None:
            candidates.append(('recovered-golfer-roi', self.initial_golfer_box))
        height, width = frame.shape[:2]
        tried = []
        for source, box in candidates:
            box = np.array([max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])], float)
            if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                continue
            xy, confidence, matrix = self.infer_crop(frame, box)
            in_frame = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                        & (xy[:, 1] >= 0) & (xy[:, 1] < height))
            valid = in_frame & np.isfinite(confidence) & (confidence >= self.confidence_threshold)
            retry = dict(valid=valid)
            tried.append(dict(source=source, bbox_xyxy=box.tolist(), body_joints=int(valid[:17].sum()),
                              usable=self.body_usable(retry)))
            if self.body_usable(retry) and valid[:17].sum() > detected['valid'][:17].sum():
                body = slice(0, 17)
                detected['points'][body] = xy[body]
                detected['valid'][body] = valid[body]
                detected['keypoint_confidence'] = detected['keypoint_confidence'].copy()
                detected['keypoint_confidence'][body] = confidence[body]
                detected['found'] = True
                good = detected['points'][:17][valid[:17]]
                detected['box'] = np.concatenate([good.min(axis=0), good.max(axis=0)])
                return dict(reason='incomplete skeleton in the golfer crop', kept=source, tried=tried)
        return dict(reason='incomplete skeleton in the golfer crop', kept=None, tried=tried)

    def fuse_body(self, frame, detected):
        """Fill body joints that ViTPose scores below 0.3 from YOLO26n inside the same crop.

        Only joints with a one-to-one COCO keypoint (shoulders, elbows, wrists,
        hips, knees, ankles) are replaced, and only where YOLO26n scores >= 0.5.
        """
        from golf_workflows.golf_pose_3d import DIRECT_FROM_COCO, JOINT
        model = getattr(self, 'body_fallback_model', None)
        crop = detected.get('pose_backend_details', {}).get('crop_bbox_xyxy')
        if model is None or crop is None:
            return []
        confidence = detected['keypoint_confidence']
        weak = [name for name in DIRECT_FROM_COCO if confidence[JOINT[name]] < FUSION_VITPOSE_BELOW]
        if not weak:
            return []
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = crop
        pad_x, pad_y = .1 * (x1 - x0), .1 * (y1 - y0)
        x0, y0 = int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y))
        x1, y1 = int(min(width, x1 + pad_x)), int(min(height, y1 + pad_y))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        result = model.predict(frame[y0:y1, x0:x1], imgsz=640, verbose=False)[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        person = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        xy = result.keypoints.xy[person].cpu().numpy().astype(np.float64) + np.array([x0, y0])
        scores = (result.keypoints.conf[person].cpu().numpy() if result.keypoints.conf is not None
                  else np.zeros(len(xy)))
        replaced = []
        detected['keypoint_confidence'] = detected['keypoint_confidence'].copy()
        for name in weak:
            joint, coco = JOINT[name], DIRECT_FROM_COCO[name]
            if scores[coco] >= FUSION_YOLO_ABOVE:
                detected['points'][joint] = xy[coco]
                detected['valid'][joint] = True
                detected['keypoint_confidence'][joint] = scores[coco]
                replaced.append(name)
        return replaced

    @torch.inference_mode()
    def predict_frame(self, frame, previous_box=None, allow_panel_fallback=True, previous_box_source='detector'):
        height, width = frame.shape[:2]
        crop_box, detector_details = self.detector.golfer_box(frame, previous=previous_box)
        crop_source = 'official-yolox-1cls'
        if crop_box is None and previous_box is not None and np.isfinite(previous_box).all():
            crop_box = np.asarray(previous_box, dtype=float).copy()
            crop_source = {'manual': 'manual-golfer-roi', 'auto-yolo26n': 'auto-golfer-roi'}.get(
                previous_box_source, 'tracked-detector-roi')
            detector_details['tracked_golfer_bbox_xyxy'] = crop_box.tolist()
            detector_details['fallback_reason'] = {
                'manual': 'current detector miss; use explicit initial golfer ROI',
                'auto-yolo26n': 'current detector miss; use automatic YOLO26n golfer ROI',
            }.get(previous_box_source, 'current detector miss; retain last detected golfer ROI')
        if crop_box is None and allow_panel_fallback:
            crop_box = np.array([0, 0, width, height], dtype=float)
            crop_source = 'reference-panel-fallback'
            detector_details['fallback_reason'] = 'no golfer detection with score >= 0.25'
        if crop_box is None:
            detector_details['crop_source'] = 'detector-miss'
            detector_details['fallback_reason'] = 'no golfer detection with score >= 0.25; video pose unobserved'
            details = dict(crop_source='detector-miss', detector_crop=detector_details,
                confidence=[0.0]*22, landmark_xy=np.full((22, 2), np.nan).tolist())
            return dict(points=np.full((22, 2), np.nan), valid=np.zeros(22, bool),
                box=np.full(4, np.nan), grip=None, head=None, found=False,
                keypoint_confidence=np.zeros(22), club_confidence=np.zeros(5),
                pose_backend_details=details, club_pose_details=details,
                club_crop_details=detector_details, detector_details=detector_details, detector_box=None)
        started = perf_counter()
        xy, confidence, matrix = self.infer_crop(frame, crop_box)
        in_frame = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                    & (xy[:, 1] >= 0) & (xy[:, 1] < height))
        valid = in_frame & np.isfinite(confidence) & (confidence >= self.confidence_threshold)
        body = xy[:17][valid[:17]]
        box = np.concatenate([body.min(axis=0), body.max(axis=0)]) if len(body) >= 2 else np.full(4, np.nan)
        detector_details['crop_source'] = crop_source
        detector_details['pose_input_bbox_xyxy'] = crop_box.tolist()
        details = dict(crop_affine=matrix.tolist(), crop_source=crop_source,
            crop_bbox_xyxy=crop_box.tolist(), detector_crop=detector_details,
            confidence=confidence.tolist(), landmark_xy=xy.tolist(), in_frame=in_frame.tolist(),
            inference_seconds=round(perf_counter() - started, 3))
        return dict(points=xy.astype(np.float64), valid=valid, box=box,
            grip=xy[17] if valid[17] else None, head=xy[18] if valid[18] else None,
            found=bool(valid[:17].any()), pose_backend_details=details,
            club_confidence=confidence[17:22], keypoint_confidence=confidence,
            club_pose_details=details, club_crop_details=detector_details,
            detector_details=detector_details,
            detector_box=crop_box if crop_source == 'official-yolox-1cls' else None)

    def detect_sequence(self, video, track_club_path=False):
        from golf_workflows.golf_pose_3d import Swing2D, fill_gaps
        from golf_workflows.trace_swing_plane import stabilize
        capture = cv2.VideoCapture(str(video))
        points, flags, boxes, confidences, wrists, samples = [], [], [], [], [], []
        detector_rows = []
        from golf_workflows.club_tracking import ShaftPredictor
        from golf_workflows.swing_phases import RELIABLE_CLUB_CONFIDENCE
        predictor = ShaftPredictor(RELIABLE_CLUB_CONFIDENCE)
        recrops, fused, fused_frames = [], {}, set()
        last_body_box = None
        previous_box = getattr(self, 'initial_golfer_box', None)
        previous_box_source = (self.metadata.get('initial_roi_source') or 'manual') if previous_box is not None else 'detector'
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        size = (0, 0)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if not points and previous_box is not None:
                    if previous_box[2] > frame.shape[1] or previous_box[3] > frame.shape[0]:
                        raise ValueError(f'--golfer-box must fit the decoded frame ({frame.shape[1]} x {frame.shape[0]} pixels)')
                detected = self.predict_frame(frame, previous_box=previous_box, allow_panel_fallback=False,
                    previous_box_source=previous_box_source)
                index = len(points)
                recovery = self.recover_body(frame, detected, last_body_box)
                if recovery is not None and detected.get('detector_details') is not None:
                    detected['detector_details']['skeleton_recovery'] = recovery
                    if recovery['kept']:
                        detected['detector_details']['crop_source'] = recovery['kept']
                        # Steer the next detector selection back to the golfer.
                        detected['detector_box'] = None
                        previous_box = detected['box'].copy()
                        previous_box_source = 'detector'
                if self.body_usable(detected):
                    last_body_box = detected['box'].copy()
                for name in self.fuse_body(frame, detected):
                    fused[name] = fused.get(name, 0) + 1
                    fused_frames.add(index)
                recrop = self.refine_club(frame, detected, predictor, index)
                if recrop is not None:
                    recrops.append(recrop)
                predictor.observe(index, detected['points'][17], detected['points'][18],
                                  float(np.min(detected['keypoint_confidence'][17:19])))
                if detected.get('detector_box') is not None:
                    previous_box = detected['detector_box']
                    previous_box_source = 'detector'
                details = detected.get('detector_details')
                detector_rows.append(dict(details, frame_index=len(points)) if details is not None else None)
                index = len(points)
                size = (frame.shape[1], frame.shape[0])
                points.append(detected['points'])
                flags.append(detected['valid'])
                boxes.append(detected['box'])
                confidences.append(detected['keypoint_confidence'])
                wrist_indices = [i for i in (13, 16) if detected['valid'][i]]
                wrists.append(float(detected['points'][wrist_indices, 1].mean()) if wrist_indices else np.nan)
                if track_club_path and detected['head'] is not None:
                    sample = stabilize(detected['grip'], detected['head'],
                        float(detected['club_confidence'][:2].mean()), index,
                        samples[-1] if samples else None, float(np.hypot(*frame.shape[:2])))
                    if sample is not None:
                        samples.append(sample)
                if index % 10 == 0:
                    print(f'  2D joint GolfPose frame {index}', flush=True)
        finally:
            capture.release()
        if not points:
            raise ValueError(f'No frames decoded from {video}')
        points, flags = np.stack(points), np.stack(flags)
        confidence = np.stack(confidences)
        if not flags[:, 0].any():
            # A model can run and return nonfinite coordinates; that is distinct
            # from never receiving a crop because the detector missed.
            pose_frames = sum(
                row['crop_source'] != 'detector-miss' if row is not None and 'crop_source' in row
                else bool(np.isfinite(frame_points).any())
                for row, frame_points in zip(detector_rows, points))
            root_peak = float(np.max(np.nan_to_num(confidence[:, 0], nan=0)))
            diagnostics = dict(source=str(video), decoded_frames=len(points), pose_inference_frames=pose_frames,
                observed_root_frames=0, root_score_threshold=self.confidence_threshold, max_root_confidence=root_peak,
                models=getattr(self, 'metadata', {}), detector_crops=detector_rows)
            diagnostic_path = None
            if getattr(self, 'diagnostics_dir', None) is not None:
                folder = Path(self.diagnostics_dir)
                folder.mkdir(parents=True, exist_ok=True)
                diagnostic_path = folder / 'pose_inference_diagnostics.json'
                diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
                np.savez_compressed(folder / 'pose_inference_diagnostics.npz', keypoints_2d=points,
                    keypoints_2d_valid=flags, keypoints_2d_confidence=confidence)
            if pose_frames == 0:
                message = (f'One-class YOLOX found no usable golfer ROI in {len(points)} frames of {video}; '
                    'ViTPose inference was skipped, and no automatic YOLO26n golfer ROI was available. '
                    'Supply --golfer-box X1 Y1 X2 Y2 in original-frame pixels to initialize tracking.')
            else:
                message = (f'ViTPose ran on {pose_frames}/{len(points)} ROI crops in {video}, but no root joint passed '
                    f'confidence >= {self.confidence_threshold} and image bounds (highest root confidence {root_peak:.3f}). '
                    'Check the selected golfer ROI.')
            if diagnostic_path is not None:
                message += f' Diagnostics: {diagnostic_path}'
            raise ValueError(message)
        return Swing2D(frames=len(points), fps=fps, size=size, points=points, valid=flags,
            filled=fill_gaps(points, flags), boxes=np.stack(boxes), samples=samples,
            wrist_y=np.asarray(wrists), club_confidence=confidence[:, 17:22],
            keypoint_confidence=confidence, detector_details=detector_rows,
            predicted_shaft_crops=recrops,
            body_fusion=dict(enabled=getattr(self, 'body_fallback_model', None) is not None,
                             vitpose_below=FUSION_VITPOSE_BELOW, yolo26n_at_least=FUSION_YOLO_ABOVE,
                             joints_replaced=fused, frames=len(fused_frames)))
