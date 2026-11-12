"""COCO adapter; full whole-body predictions follow the winning recovery crop."""
import cv2
import numpy as np
import torch

from golf_workflows.detector_pose import YOLOXBodyPose
from .runtime import REVISION


def extract_prediction(instances, offset=(0, 0)):
    keypoints = np.asarray(instances.keypoints, dtype=np.float32)
    if keypoints.shape != (1, 133, 2):
        raise ValueError(f'Expected one 133-point 2D prediction, got {keypoints.shape}')
    xy = keypoints[0].copy()
    scores = np.asarray(instances.keypoint_scores, dtype=np.float32).reshape(-1, 133)[0].copy()
    xy += np.asarray(offset, dtype=np.float32)
    scores[~np.isfinite(xy).all(axis=1) | ~np.isfinite(scores)] = 0
    return xy, scores


class RTMWBodyPose(YOLOXBodyPose):
    def __init__(self, model, detector, checkpoint, threads=8):
        from types import SimpleNamespace
        super().__init__(SimpleNamespace(metadata=dict(pose_backend='rtmw-x-2d',
            pose_checkpoint=str(checkpoint), upstream_revision=REVISION,
            strict_parameter_load=True, phase_3d_backend='MixSTE-17+2')), detector)
        self.model = model
        metadata = getattr(model, 'workflow_metadata', {})
        if isinstance(metadata, dict):
            self.metadata.update(metadata)
        self.cpu_threads = threads
        self.metadata['cpu_threads'] = threads
        self.metadata['detector_miss_policy'] = 'tracked/manual/automatic ROI; otherwise missing'
        self.records = []
        self.replay_results = None
        self._crops = {}

    def _pose_in(self, frame, bounds, **kwargs):
        from .official import inference
        x0, y0, x1, y1 = bounds
        crop = frame[y0:y1, x0:x1]
        with torch.inference_mode():
            results = inference(self.model, crop,
                bboxes=np.array([[0, 0, x1-x0, y1-y0]], dtype=np.float32))
        if not results:
            return {0: 'person'}, torch.empty((0, 6)), torch.empty((0, 17, 3))
        xy, scores = extract_prediction(results[0].pred_instances, (x0, y0))
        self._crops[tuple(bounds)] = (xy, scores)
        boxes = torch.tensor([[x0, y0, x1, y1, float(scores[:17].mean()), 0]])
        points = torch.from_numpy(np.concatenate([xy[:17], scores[:17, None]], axis=1)[None])
        return {0: 'person'}, boxes, points

    def predict_frame(self, frame, previous=None, **kwargs):
        self._crops = {}
        result, box = super().predict_frame(frame, previous, **kwargs)
        bounds = tuple(result.detector_details['body_crop_bbox_xyxy'])
        prediction = self._crops.get(bounds)
        # A top-down estimator always returns a pose even on an empty frame.
        # Do not treat an unanchored full-frame prediction as a detected person.
        if result.detector_details['selected_person_bbox_xyxy'] is None and self.initial_golfer_box is None and previous is None:
            prediction = None
            result.update(boxes=torch.empty((0, 6)), keypoints=torch.empty((0, 17, 3)))
            result.detector_details['skeleton_usable'] = False
            self.last_body_box = None
        self.current = prediction
        return result, box

    def predict(self, source, stream=False, **kwargs):
        if self.replay_results is not None:
            return iter(self.replay_results) if stream else list(self.replay_results)
        def iterate():
            # Ultralytics' initial ROI predictor may reset Torch's thread count.
            # Reapply the requested RTMW budget after that predictor has run.
            torch.set_num_threads(self.cpu_threads)
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {source}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                capture.release()
                raise ValueError('Video has no valid FPS')
            self.predictor.dataset.fps = float(fps)
            previous = self.initial_golfer_box
            self.last_body_box = None
            self.records = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    index = len(self.records)
                    raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    time_ms = raw_ms
                    fallback = not np.isfinite(time_ms) or time_ms < 0 or (index > 0 and time_ms <= self.records[-1]['time_ms'])
                    if fallback:
                        time_ms = (self.records[-1]['time_ms'] + 1000/fps) if index else 0.
                    result, box = self.predict_frame(frame, previous, **kwargs)
                    if box is not None:
                        previous = box
                    self.records.append(dict(frame_index=index, time_ms=time_ms,
                        timestamp_fallback=fallback, prediction=self.current,
                        detector=result.detector_details))
                    if index % 50 == 0:
                        print(f'RTMW frame {index}', flush=True)
                    yield result
            finally:
                capture.release()
        generated = iterate()
        return generated if stream else list(generated)

    def save_raw(self, folder):
        from .report import write_json
        n = len(self.records)
        xy = np.full((n, 133, 2), np.nan, np.float32)
        scores = np.zeros((n, 133), np.float32)
        for i, row in enumerate(self.records):
            if row['prediction'] is not None:
                xy[i], scores[i] = row['prediction']
        np.savez_compressed(folder / 'wholebody.npz', image_2d=xy, confidence=scores,
            frame_index=np.arange(n),
            joints=np.array([self.model.dataset_meta['keypoint_id2name'][i] for i in range(133)])
                if hasattr(self.model, 'dataset_meta') and isinstance(self.model.dataset_meta, dict) else np.arange(133),
            times_ms=np.array([r['time_ms'] for r in self.records]),
            timestamp_fallback=np.array([r['timestamp_fallback'] for r in self.records]))
        write_json(folder / 'wholebody.json', dict(schema_version=2, model=self.metadata, frames=n,
            image_coordinates='original frame pixels; x right, y down',
            native_3d_available=False,
            quality=dict(missing_frames=int(np.sum(~np.isfinite(xy).any(axis=(1, 2)))),
                         mean_body_confidence=float(scores[:, :17].mean()) if n else None,
                         body_points_below_025_per_frame=(scores[:, :17] < .25).sum(axis=1).tolist()),
            phase_coordinates='COCO17 -> existing GolfPose mapping -> MixSTE17+2',
            detections=[r['detector'] for r in self.records]))
