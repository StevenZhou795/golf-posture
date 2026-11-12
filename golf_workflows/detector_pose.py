"""YOLOX person crops for YOLO26n pose; all returned coordinates remain image-space."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from golf_workflows.official_yolox import select_box

BODY_KEYPOINT_CONFIDENCE = 0.25
BODY_KEYPOINTS_REQUIRED = 12


class YOLOXBodyPose:
    def __init__(self, pose_model, detector):
        self.pose_model = pose_model
        self.detector = detector
        self.predictor = SimpleNamespace(dataset=SimpleNamespace(fps=30.0))
        self.initial_golfer_box = None
        self.last_body_box = None
        self.last_body_keypoints = None
        self.lock_anchor = None
        self.metadata = dict(pose_model.metadata, person_detector=detector.metadata,
            body_crop_backend='official-yolox-2cls', body_crop_expansion=1.10,
            locked_crop_padding_fraction=.12, locked_crop_min_padding_height=.08,
            detector_miss_policy='retain confirmed target ROI; never reacquire another person after locking',
            person_selection_policy='geometry continuity plus skeleton identity gate')

    def reset_tracking(self):
        self.last_body_box = None
        self.last_body_keypoints = None
        self.lock_anchor = None

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
        self.reset_tracking()

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

    def _target_pose(self, boxes, keypoints):
        """Keep one identity; a confident bystander or mixed skeleton is a miss."""
        keypoints = keypoints.clone()
        keypoints[..., 2][~torch.isfinite(keypoints).all(dim=-1)] = 0
        candidates = []
        for index, points in enumerate(keypoints.numpy()):
            good = (points[:, 2] >= BODY_KEYPOINT_CONFIDENCE) & np.isfinite(points).all(axis=1)
            if not good[[5, 6, 11, 12]].all():
                continue
            height = max(float(np.ptp(points[good, 1])), 1.)
            if height < 8:
                continue
            penalty = 0.
            if self.last_body_keypoints is not None:
                old = self.last_body_keypoints
                common = good & (old[:, 2] >= BODY_KEYPOINT_CONFIDENCE)
                scale = max(float(self.last_body_box[3] - self.last_body_box[1]), 1.)
                movement = np.linalg.norm(points[:, :2] - old[:, :2], axis=1) / scale
                # Check the head/shoulders separately: their jump can be hidden
                # by stationary legs in a mixed golfer/spectator skeleton.
                core = np.array([0, 5, 6, 11, 12])
                core = core[common[core]]
                if (len(core) < 4 or np.median(movement[common]) > .20
                        or np.max(movement[core]) > .30):
                    continue
                penalty = float(np.mean(movement[core]))
                # An immutable lower-body anchor prevents small accepted steps
                # from gradually transferring the track to a background person.
                hips = points[[11, 12], :2].mean(axis=0)
                if np.linalg.norm(hips - self.lock_anchor[:2]) > .35 * self.lock_anchor[2]:
                    continue
            score = float(points[good, 2].sum()) - 10 * penalty
            if self.last_body_keypoints is None and self.initial_golfer_box is not None:
                roi = self.initial_golfer_box
                extent = np.r_[points[good, :2].min(axis=0), points[good, :2].max(axis=0)]
                intersection = np.maximum(0, np.minimum(extent[2:], roi[2:])
                                          - np.maximum(extent[:2], roi[:2])).prod()
                union = np.prod(extent[2:] - extent[:2]) + np.prod(roi[2:] - roi[:2]) - intersection
                score += 17 * intersection / max(union, 1.)
            candidates.append((score, index))
        if not candidates:
            return boxes[:0], keypoints[:0]
        index = max(candidates)[1]
        return boxes[index:index + 1], keypoints[index:index + 1]

    def _tracked_crop(self):
        """Pad the confirmed body once, never expand a previous padded crop."""
        last = self.last_body_box
        size = last[2:] - last[:2]
        margin = np.maximum(size * .12, size[1] * .08)
        return np.r_[last[:2] - margin, last[2:] + margin]

    def predict_frame(self, frame, previous=None, **kwargs):
        from ultralytics.engine.results import Results
        detection = self.detector.predict(frame)
        reference = self.last_body_box if self.last_body_box is not None else previous
        if reference is None:
            reference = self.initial_golfer_box
        box, score = select_box(detection, minimum_score=.25, previous=reference, require_overlap=True)
        height, width = frame.shape[:2]
        bounds = np.array([0, 0, width, height], dtype=int)
        crop_source = 'full-frame-fallback'
        if self.last_body_box is not None:
            bounds = self._clip_bounds(self._tracked_crop(), width, height)
            crop_source = 'locked-body-roi'
        elif self.initial_golfer_box is not None:
            clipped = self._clip_bounds(self.initial_golfer_box, width, height)
            if clipped is not None:
                bounds = clipped
                crop_source = 'auto-golfer-roi' if self.metadata.get('initial_roi_source') == 'auto-yolo26n' else 'manual-golfer-roi'
        elif box is not None:
            center = (box[:2] + box[2:]) / 2
            half = (box[2:] - box[:2]) * .55
            clipped = self._clip_bounds(np.r_[center - half, center + half], width, height)
            if clipped is not None:
                bounds, crop_source = clipped, 'yolox-person'
        names, boxes, keypoints = self._pose_in(frame, bounds, **kwargs)
        boxes, keypoints = self._target_pose(boxes, keypoints)
        usable, quality = self._quality(keypoints)
        recovery = None
        if not usable:
            # The person box can slide onto spectators or background while the
            # golfer moves fast (downswing, finish). A detector retry must pass
            # the same identity gate before it can replace the locked crop.
            candidates = []
            if box is not None:
                center = (box[:2] + box[2:]) / 2
                half = (box[2:] - box[:2]) * .6
                candidates.append(('recovered-detector-roi', np.r_[center - half, center + half]))
            if self.last_body_box is None and self.initial_golfer_box is not None:
                candidates.append(('recovered-golfer-roi', self.initial_golfer_box))
            tried = []
            for source, values in candidates:
                retry_bounds = self._clip_bounds(np.asarray(values, float), width, height)
                if retry_bounds is None or np.array_equal(retry_bounds, bounds):
                    continue
                retry = self._pose_in(frame, retry_bounds, **kwargs)
                retry_boxes, retry_points = self._target_pose(retry[1], retry[2])
                retry = (retry[0], retry_boxes, retry_points)
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
            self.last_body_keypoints = keypoints[best].numpy().copy()
            if self.lock_anchor is None:
                hips = self.last_body_keypoints[[11, 12], :2].mean(axis=0)
                self.lock_anchor = np.r_[hips, self.last_body_box[3] - self.last_body_box[1]]
        elif self.last_body_box is not None:
            # Keep the identity state, but expose missing data instead of a
            # rejected pose that downstream smoothing could turn into a switch.
            boxes, keypoints = boxes[:0], keypoints[:0]
        result = Results(frame, path='', names=names, boxes=boxes, keypoints=keypoints)
        result.yolox_detection = detection  # Club crop reuses this detector pass.
        result.detector_details = dict(detector=self.detector.metadata,
            selected_person_bbox_xyxy=None if box is None else box.tolist(),
            selected_person_score=score, body_crop_bbox_xyxy=bounds.tolist(),
            body_crop_source=crop_source, skeleton_usable=usable, skeleton_recovery=recovery,
            target_locked=self.lock_anchor is not None,
            tracking_status='confirmed' if usable else 'missing-hold-target',
            fallback_reason=None if box is not None else 'no identity-compatible person detection')
        if len(keypoints):
            # Existing reference ranking consumes the 17 COCO confidence scores.
            areas = (boxes[:, 2:4] - boxes[:, :2]).prod(dim=1)
            chosen = int(areas.argmax())
            result.vitpose_details = dict(confidence=keypoints[chosen, :, 2].tolist(),
                detector_crop=result.detector_details)
        if self.last_body_box is not None:
            # Steer the next YOLOX selection back to the golfer, not the drifted
            # or stale box.
            return result, self.last_body_box.copy()
        return result, previous

    def predict(self, source, stream=False, **kwargs):
        def iterate():
            self.reset_tracking()
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
