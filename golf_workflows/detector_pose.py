"""YOLOX person crops for YOLO26n pose; all returned coordinates remain image-space."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from golf_workflows.official_yolox import select_box

BODY_KEYPOINT_CONFIDENCE = 0.25
BODY_KEYPOINTS_REQUIRED = 12
# Retry crop around the last complete skeleton, as a multiple of its extent.
RECOVERY_EXPANSION = 1.6


class YOLOXBodyPose:
    def __init__(self, pose_model, detector):
        self.pose_model = pose_model
        self.detector = detector
        self.predictor = SimpleNamespace(dataset=SimpleNamespace(fps=30.0))
        self.initial_golfer_box = None
        self.last_body_box = None
        self.metadata = dict(pose_model.metadata, person_detector=detector.metadata,
            body_crop_backend='official-yolox-2cls', body_crop_expansion=1.10,
            detector_miss_policy='golfer ROI (manual or automatic YOLO26n) if set, else YOLO26n full frame; recorded per frame')

    def set_initial_golfer_box(self, values, source='manual', details=None):
        """Crop used on YOLOX person misses, so the largest person in the frame
        cannot be a spectator outside the golfer's area."""
        if values is None:
            self.initial_golfer_box = None
        else:
            from golf_workflows.golfer_vitpose import validate_golfer_box
            self.initial_golfer_box = validate_golfer_box(values)
        self.metadata['initial_golfer_box_xyxy'] = None if values is None else self.initial_golfer_box.tolist()
        self.metadata['initial_roi_source'] = None if values is None else source
        self.metadata['initial_roi_details'] = details

    @staticmethod
    def _clip_bounds(values, width, height):
        bounds = np.r_[np.floor(values[:2] + 1e-6), np.ceil(values[2:] - 1e-6)].astype(int)
        bounds[[0, 2]] = bounds[[0, 2]].clip(0, width)
        bounds[[1, 3]] = bounds[[1, 3]].clip(0, height)
        return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None

    def _pose_in(self, frame, bounds, **kwargs):
        x0, y0, x1, y1 = bounds
        pose = self.pose_model.predict(frame[y0:y1, x0:x1], stream=False, **kwargs)[0]
        boxes = pose.boxes.data.detach().cpu().clone() if pose.boxes is not None else torch.empty((0, 6))
        keypoints = (pose.keypoints.data.detach().cpu().clone() if pose.keypoints is not None
                     else torch.empty((0, 17, 3)))
        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0
        keypoints[..., :2] += torch.tensor([x0, y0])
        return pose.names, boxes, keypoints

    @staticmethod
    def _quality(keypoints):
        """(usable, score) for the largest-confidence skeleton in a crop.

        Usable needs at least BODY_KEYPOINTS_REQUIRED confident COCO keypoints
        including both shoulders and a wrist; a crop that slid onto the legs or
        onto background fails this even when the detector box score is high.
        """
        if not len(keypoints):
            return False, 0.0
        confidence = keypoints[..., 2].numpy()
        best = int(confidence.sum(axis=1).argmax())
        good = confidence[best] >= BODY_KEYPOINT_CONFIDENCE
        usable = (good.sum() >= BODY_KEYPOINTS_REQUIRED and good[5] and good[6] and (good[9] or good[10]))
        return bool(usable), float(confidence[best].sum())

    def predict_frame(self, frame, previous=None, **kwargs):
        from ultralytics.engine.results import Results
        detection = self.detector.predict(frame)
        box, score = select_box(detection, minimum_score=.25, previous=previous, require_overlap=True)
        height, width = frame.shape[:2]
        bounds = np.array([0, 0, width, height], dtype=int)
        crop_source = 'full-frame-fallback'
        if box is not None:
            center = (box[:2] + box[2:]) / 2
            half = (box[2:] - box[:2]) * .55
            clipped = self._clip_bounds(np.r_[center - half, center + half], width, height)
            if clipped is not None:
                bounds, crop_source = clipped, 'yolox-person'
        elif self.initial_golfer_box is not None:
            clipped = self._clip_bounds(self.initial_golfer_box, width, height)
            if clipped is not None:
                bounds = clipped
                crop_source = 'auto-golfer-roi' if self.metadata.get('initial_roi_source') == 'auto-yolo26n' else 'manual-golfer-roi'
        names, boxes, keypoints = self._pose_in(frame, bounds, **kwargs)
        usable, quality = self._quality(keypoints)
        recovery = None
        if not usable:
            # The person box can slide onto spectators or background while the
            # golfer moves fast (downswing, finish). Retry around the last complete
            # skeleton and inside the golfer ROI; keep whichever skeleton is best.
            candidates = []
            if getattr(self, 'last_body_box', None) is not None:
                last = self.last_body_box
                center, half = (last[:2] + last[2:]) / 2, (last[2:] - last[:2]) * .5 * RECOVERY_EXPANSION
                candidates.append(('recovered-last-skeleton', np.r_[center - half, center + half]))
            if self.initial_golfer_box is not None:
                candidates.append(('recovered-golfer-roi', self.initial_golfer_box))
            tried = []
            for source, values in candidates:
                retry_bounds = self._clip_bounds(np.asarray(values, float), width, height)
                if retry_bounds is None:
                    continue
                retry = self._pose_in(frame, retry_bounds, **kwargs)
                retry_usable, retry_quality = self._quality(retry[2])
                tried.append(dict(source=source, bbox_xyxy=retry_bounds.tolist(), usable=retry_usable,
                                  confidence_sum=round(retry_quality, 2)))
                if (retry_usable, retry_quality) > (usable, quality):
                    names, boxes, keypoints = retry
                    usable, quality, bounds, crop_source = retry_usable, retry_quality, retry_bounds, source
            recovery = dict(reason='incomplete skeleton in the first crop', tried=tried)
        if usable:
            confidence = keypoints[..., 2].numpy()
            best = int(confidence.sum(axis=1).argmax())
            good = confidence[best] >= BODY_KEYPOINT_CONFIDENCE
            xy = keypoints[best, good, :2].numpy()
            self.last_body_box = np.r_[xy.min(axis=0), xy.max(axis=0)]
        result = Results(frame, path='', names=names, boxes=boxes, keypoints=keypoints)
        result.yolox_detection = detection  # Club crop reuses this detector pass.
        result.detector_details = dict(detector=self.detector.metadata,
            selected_person_bbox_xyxy=None if box is None else box.tolist(),
            selected_person_score=score, body_crop_bbox_xyxy=bounds.tolist(),
            body_crop_source=crop_source, skeleton_usable=usable, skeleton_recovery=recovery,
            fallback_reason=None if box is not None else 'no person detection with score >= 0.25')
        if len(keypoints):
            # Existing reference ranking consumes the 17 COCO confidence scores.
            areas = (boxes[:, 2:4] - boxes[:, :2]).prod(dim=1)
            chosen = int(areas.argmax())
            result.vitpose_details = dict(confidence=keypoints[chosen, :, 2].tolist(),
                detector_crop=result.detector_details)
        if usable and crop_source not in ('yolox-person', 'full-frame-fallback'):
            # Steer the next YOLOX selection back to the golfer, not the drifted
            # or stale box.
            return result, self.last_body_box.copy()
        return result, box

    def predict(self, source, stream=False, **kwargs):
        def iterate():
            if isinstance(source, np.ndarray):
                yield self.predict_frame(source, **kwargs)[0]
                return
            capture = cv2.VideoCapture(str(Path(source)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.predictor.dataset.fps = fps if np.isfinite(fps) and fps > 0 else 30.0
            # Start from the golfer ROI: the highest-scoring person on the first
            # frame can be a bystander (009: guard 0.950, golfer 0.939).
            previous = None if self.initial_golfer_box is None else self.initial_golfer_box.copy()
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    result, box = self.predict_frame(frame, previous, **kwargs)
                    previous = box if box is not None else previous
                    yield result
            finally:
                capture.release()
        generated = iterate()
        return generated if stream else list(generated)
