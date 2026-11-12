import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from rtmw_workflow.overlay import draw_detections
from rtmw_workflow.report import render_wholebody


class OverlayTests(unittest.TestCase):
    def test_all_three_native_classes_have_semantic_labels_and_confidence(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        rows = [dict(class_id=i, box=[30+i*180, 100, 130+i*180, 250], confidence=.87)
                for i in range(3)]
        with patch('rtmw_workflow.overlay.cv2.putText', wraps=cv2.putText) as put:
            returned = draw_detections(frame, dict(detections=rows))
        self.assertIs(returned, frame)
        self.assertEqual([call.args[1] for call in put.call_args_list],
                         ['shaft 0.87', 'head 0.87', 'hands/grip 0.87'])
        self.assertTrue(frame[200, 30].any())
        self.assertTrue(frame[200, 210].any())
        self.assertTrue(frame[200, 390].any())

    def test_missing_and_invalid_boxes_are_not_drawn_or_carried_forward(self):
        invalid = [dict(class_id=0, box=[float('nan'), 0, 10, 10], confidence=.8),
                   dict(class_id=1, box=[-40, 10, -20, 20], confidence=.8),
                   dict(class_id=8, box=[10, 10, 20, 20], confidence=.8)]
        for row in (None, dict(detections=[]), dict(detections=invalid)):
            frame = np.zeros((100, 100, 3), np.uint8)
            draw_detections(frame, row)
            self.assertFalse(frame.any())

    def test_render_reads_saved_frame_indices_without_reusing_previous_boxes(self):
        capture, writer = Mock(), Mock()
        capture.get.side_effect = lambda key: {cv2.CAP_PROP_FPS:30, 3:320, 4:240}[key]
        capture.read.side_effect = [(True, np.zeros((240, 320, 3), np.uint8)) for _ in range(2)]
        writer.isOpened.return_value = True
        frames = []
        writer.write.side_effect = lambda frame: frames.append(frame.copy())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.savez(folder/'wholebody.npz', image_2d=np.zeros((2, 133, 2)), confidence=np.zeros((2, 133)))
            (folder/'swingtrace_detections.json').write_text(json.dumps(dict(frames=[
                dict(frame_index=1, detections=[dict(class_id=1, box=[100, 150, 200, 200], confidence=.9)])])))
            with patch('rtmw_workflow.report.cv2.VideoCapture', return_value=capture), \
                 patch('rtmw_workflow.report.cv2.VideoWriter', return_value=writer):
                render_wholebody(folder, Path('synthetic-placeholder'), SimpleNamespace(dataset_meta={}))
        self.assertEqual(len(frames), 2)
        self.assertFalse(frames[0][200, 150].any())
        self.assertTrue(frames[1][200, 150].any())
        capture.release.assert_called_once()
        writer.release.assert_called_once()
