"""Regression cases for the mixed golfer/spectator skeleton seen in 008."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from golf_workflows.detector_pose import YOLOXBodyPose
from golf_workflows.official_yolox import select_box


def skeleton():
    xy = [[150, 60], [145, 58], [155, 58], [140, 62], [160, 62],
          [130, 90], [170, 90], [125, 130], [175, 130], [140, 160],
          [160, 160], [140, 170], [160, 170], [135, 220], [165, 220],
          [130, 270], [170, 270]]
    return torch.tensor([[list(p) + [.9] for p in xy]])


class PersonLockTests(unittest.TestCase):
    def body(self):
        detector = Mock(metadata={})
        detector.predict.return_value = dict(boxes=np.array([[110, 40, 190, 280]]),
            scores=np.array([.8]), labels=np.array([0]))
        body = YOLOXBodyPose(SimpleNamespace(metadata={}), detector)
        body._pose_in = Mock(return_value=({0: 'person'},
            torch.tensor([[110., 40., 190., 280., .9, 0.]]), skeleton()))
        return body

    def test_overlap_continuity_beats_bystander_confidence(self):
        result = dict(boxes=np.array([[102, 50, 202, 300], [155, 50, 255, 300]]),
            scores=np.array([.6, .99]), labels=np.array([0, 0]))
        box, _ = select_box(result, previous=np.array([100, 50, 200, 300]), require_overlap=True)
        np.testing.assert_array_equal(box, result['boxes'][0])

    def test_detector_gap_retains_crop_and_identity(self):
        body = self.body()
        frame = np.zeros((360, 400, 3), np.uint8)
        body.predict_frame(frame)
        last = body.last_body_box.copy()
        body.detector.predict.return_value = dict(boxes=np.empty((0, 4)), scores=[], labels=[])
        result, tracked = body.predict_frame(frame)
        self.assertEqual(result.detector_details['body_crop_source'], 'locked-body-roi')
        self.assertIsNone(result.detector_details['selected_person_bbox_xyxy'])
        np.testing.assert_array_equal(tracked, last)
        self.assertTrue(result.detector_details['target_locked'])

    def test_high_confidence_mixed_skeleton_cannot_update_lock(self):
        body = self.body()
        frame = np.zeros((360, 400, 3), np.uint8)
        body.predict_frame(frame)
        last = body.last_body_box.copy()
        mixed = skeleton()
        mixed[0, :7, 0] += 90
        mixed[..., 2] = .99
        body._pose_in.return_value = ({0: 'person'},
            torch.tensor([[110., 40., 290., 280., .99, 0.]]), mixed)
        for _ in range(3):
            result, tracked = body.predict_frame(frame)
            self.assertEqual(len(result.keypoints), 0)
            self.assertEqual(result.detector_details['tracking_status'], 'missing-hold-target')
            np.testing.assert_array_equal(tracked, last)
        body._pose_in.return_value = ({0: 'person'},
            torch.tensor([[110., 40., 190., 280., .9, 0.]]), skeleton())
        result, _ = body.predict_frame(frame)
        self.assertTrue(result.detector_details['skeleton_usable'])

    def test_slow_drift_is_bounded_by_original_anchor(self):
        body = self.body()
        frame = np.zeros((360, 400, 3), np.uint8)
        body.predict_frame(frame)
        original = body.lock_anchor.copy()
        boxes = body._pose_in.return_value[1]
        for shift in (20, 40, 60):
            points = skeleton()
            points[..., 0] += shift
            body._pose_in.return_value = ({0: 'person'}, boxes, points)
            body.predict_frame(frame)
        points = skeleton()
        points[..., 0] += 80
        _, accepted = body._target_pose(boxes, points)
        self.assertEqual(len(accepted), 0)
        np.testing.assert_array_equal(body.lock_anchor, original)

    def test_reset_releases_previous_video_identity(self):
        body = self.body()
        body.predict_frame(np.zeros((360, 400, 3), np.uint8))
        body.reset_tracking()
        self.assertIsNone(body.last_body_box)
        self.assertIsNone(body.last_body_keypoints)
        self.assertIsNone(body.lock_anchor)


if __name__ == '__main__':
    unittest.main()
