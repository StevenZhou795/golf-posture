"""Render saved native detections in original image coordinates, without inference."""
import cv2
import numpy as np

# OpenCV BGR colours, shared by the video and selected-frame stills.
STYLES = {0: ('shaft', (255, 210, 0)),
          1: ('head', (180, 80, 255)),
          2: ('hands/grip', (80, 220, 80))}


def draw_detections(frame, observation):
    """Draw all native boxes for this frame in place; missing means no box."""
    height, width = frame.shape[:2]
    for detection in (observation or {}).get('detections', []):
        style = STYLES.get(detection.get('class_id'))
        box = np.asarray(detection.get('box', []), dtype=float)
        confidence = float(detection.get('confidence', float('nan')))
        if (style is None or box.shape != (4,) or not np.isfinite(box).all()
                or not np.isfinite(confidence) or not 0 <= confidence <= 1
                or np.any(box[2:] <= box[:2])):
            continue
        x1, y1, x2, y2 = np.rint(np.clip(box, [0, 0, 0, 0],
                                       [width-1, height-1, width-1, height-1])).astype(int)
        if x2 <= x1 or y2 <= y1:
            continue
        name, colour = style
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        label = f'{name} {confidence:.2f}'
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, .5
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, 1)
        x = max(0, min(x1, width-text_width-5))
        y = y1-5 if y1 >= text_height+9 else y1+text_height+5
        y = min(max(text_height+2, y), height-baseline-2)
        cv2.rectangle(frame, (x, max(0, y-text_height-3)),
                      (min(width-1, x+text_width+4), min(height-1, y+baseline+2)), (25, 25, 25), -1)
        cv2.putText(frame, label, (x+2, y), font, scale, colour, 1, cv2.LINE_AA)
    return frame
