"""Default golfer ROI from a quick YOLO26n pass, used when GolfPose detectors miss.

Sample frames across the clip, take the largest confident person in each,
keep the boxes that agree with their median (the golfer stays roughly in place
during a swing), and expand their union so the club fits: the shaft reaches
above the head at the top and finish, and out to the sides at P2/P8.
"""
from pathlib import Path

import cv2
import numpy as np

from golf_workflows.vitpose_tracking import ROOT

POSE_WEIGHTS = ROOT / 'yolo26n-pose.pt'
SAMPLES = 12
MIN_SCORE = .4
MIN_FRAMES = 3
# Margins as fractions of the union height: left/right, top, bottom.
SIDE_MARGIN, TOP_MARGIN, BOTTOM_MARGIN = .15, .25, .05


def _iou(boxes, box):
    wh = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]))
    inter = wh.prod(axis=1)
    union = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) + np.prod(box[2:] - box[:2]) - inter
    return inter / np.maximum(union, 1e-6)


def roi_from_person_boxes(boxes, frame_size):
    """Expanded union of the boxes that overlap their median by IoU >= 0.3."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) < MIN_FRAMES:
        return None, boxes[:0]
    median = np.median(boxes, axis=0)
    kept = boxes[_iou(boxes, median) >= .3]
    if len(kept) < MIN_FRAMES:
        return None, kept
    union = np.r_[kept[:, :2].min(axis=0), kept[:, 2:].max(axis=0)]
    height = union[3] - union[1]
    width, frame_height = frame_size
    roi = union + np.array([-SIDE_MARGIN, -TOP_MARGIN, SIDE_MARGIN, BOTTOM_MARGIN]) * height
    roi = np.clip(roi, 0, [width - 1, frame_height - 1, width - 1, frame_height - 1])
    if not (roi[2] > roi[0] and roi[3] > roi[1]):
        return None, kept
    return roi, kept


def auto_golfer_roi(video, model=None, samples=SAMPLES):
    """Return (roi_xyxy or None, details) for a video."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(POSE_WEIGHTS))
    capture = cv2.VideoCapture(str(Path(video)))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wanted = set(np.linspace(0, max(total - 1, 0), min(samples, max(total, 1))).round().astype(int).tolist())
    boxes, size, index = [], (0, 0), 0
    try:
        while wanted and index <= max(wanted):
            ok, frame = capture.read()
            if not ok:
                break
            size = (frame.shape[1], frame.shape[0])
            if index in wanted:
                result = model.predict(frame, imgsz=640, verbose=False)[0]
                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.cpu().numpy().astype(float)
                    score = result.boxes.conf.cpu().numpy().astype(float)
                    xyxy = xyxy[score >= MIN_SCORE]
                    if len(xyxy):
                        boxes.append(xyxy[np.argmax((xyxy[:, 2:] - xyxy[:, :2]).prod(axis=1))])
            index += 1
    finally:
        capture.release()
    roi, kept = roi_from_person_boxes(boxes, size)
    details = dict(source='auto-yolo26n', detector=POSE_WEIGHTS.name, sampled_frames=len(wanted),
                   person_frames=len(boxes), agreeing_frames=int(len(kept)),
                   margins_of_height=dict(side=SIDE_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN),
                   roi_xyxy=None if roi is None else [round(float(v), 1) for v in roi])
    return roi, details
