import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from golf_workflows.golfer_vitpose import FLIP, GolferViTPose
from golf_workflows.golf_pose_3d import detect_2d
from golf_workflows.standard_frames import detect_view


class GolferViTPoseTests(unittest.TestCase):
    def test_joint_backend_loader_does_not_construct_a_yolo_model(self):
        from golf_workflows.vitpose_tracking import load_pose_backend
        with patch('golf_workflows.golfer_vitpose.GolferViTPose') as joint, \
             patch('ultralytics.YOLO') as yolo:
            self.assertIs(load_pose_backend('golfpose-golfer-vitpose-huge', 3), joint.return_value)
            joint.assert_called_once_with(cpu_threads=3)
            yolo.assert_not_called()

    def test_public_reference_and_video_commands_default_to_yolo_body_and_club_vitpose(self):
        from golf_workflows import cli
        for command in ('references', 'video'):
            with patch.object(cli, command) as run:
                cli.main([command])
                args = run.call_args.args[0]
                self.assertEqual(args.pose_backend, 'yolo26n-pose')
                self.assertEqual(args.club_pose_backend, 'vitpose-huge')
                self.assertEqual(args.club_crop_backend, 'official-yolox-2cls')
                self.assertEqual(args.lifter, '17+2')

    def test_native_22_channels_and_body_flip_swaps_are_preserved(self):
        model = GolferViTPose.__new__(GolferViTPose)
        model.detector = Mock()
        model.detector.golfer_box.return_value = (np.array([0., 0., 192., 256.]), {})
        heatmaps = torch.zeros(2, 22, 64, 48)
        expected = []
        for index in range(22):
            x, y = 8 + index % 6 * 5, 10 + index // 6 * 9
            heatmaps[0, index, y, x] = 1
            heatmaps[1, FLIP[index], y, 47 - x] = 1
            expected.append([x / 47 * 240 - 24, y / 63 * 320 - 32])
        model.model = lambda image: heatmaps
        detected = model.predict_frame(np.zeros((256, 192, 3), np.uint8))
        np.testing.assert_allclose(detected['points'], expected, atol=.2)
        self.assertTrue(detected['valid'].all())
        self.assertEqual(len(detected['keypoint_confidence']), 22)
        np.testing.assert_array_equal(detected['grip'], detected['points'][17])
        np.testing.assert_array_equal(detected['head'], detected['points'][18])

    def test_outside_and_low_confidence_are_not_observations(self):
        model = GolferViTPose.__new__(GolferViTPose)
        model.detector = Mock()
        model.detector.golfer_box.return_value = (np.array([0., 0., 192., 256.]), {})
        model.model = lambda image: torch.zeros(2, 22, 64, 48)
        xy = np.ones((22, 2), np.float32) * 30
        confidence = np.ones(22, np.float32)
        xy[18, 0] = -10
        confidence[21] = .01
        with patch('golf_workflows.golfer_vitpose.decode_heatmaps', return_value=(xy, confidence)):
            result = model.predict_frame(np.zeros((256, 192, 3), np.uint8))
        self.assertFalse(result['valid'][18])
        self.assertFalse(result['valid'][21])
        self.assertEqual(result['points'][18, 0], -10)
        self.assertIsNone(result['head'])

    def test_reference_and_video_routes_bypass_legacy_pose_and_club_models(self):
        model = Mock(spec=['joint_pose', 'predict_frame', 'detect_sequence'])
        model.joint_pose = True
        model.predict_frame.return_value = {'points': np.ones((22, 2))}
        legacy = Mock()
        frame = np.zeros((256, 192, 3))
        self.assertIs(detect_view(frame, model, legacy, legacy, legacy), model.predict_frame.return_value)
        self.assertIs(detect_2d('clip.mp4', model, legacy, club_pose_model=legacy), model.detect_sequence.return_value)
        legacy.predict.assert_not_called()
        legacy.club_box.assert_not_called()

    def test_17plus2_lifter_drops_clubhead_outline_and_its_confidence(self):
        from types import SimpleNamespace
        from golf_workflows.golf_pose_3d import Swing2D, fill_gaps, swing_for_lifter
        points = np.random.default_rng(3).normal(size=(4, 22, 2))
        valid = np.ones((4, 22), bool)
        confidence = np.tile(np.arange(22) / 22, (4, 1))
        swing = Swing2D(frames=4, fps=30., size=(64, 64), points=points, valid=valid,
            filled=fill_gaps(points, valid), boxes=np.zeros((4, 4)), samples=[], wrist_y=np.zeros(4),
            club_confidence=confidence[:, 17:], keypoint_confidence=confidence)
        lifted = swing_for_lifter(swing, SimpleNamespace(Spatial_pos_embed=torch.zeros(1, 19, 8)))
        np.testing.assert_array_equal(lifted.points, points[:, :19])
        np.testing.assert_array_equal(lifted.keypoint_confidence, confidence[:, :19])
        np.testing.assert_array_equal(lifted.club_confidence, confidence[:, 17:19])
        self.assertEqual(lifted.filled.shape, (4, 19, 2))

    def test_17plus2_reference_club_overlay_uses_shaft_and_hosel(self):
        from golf_workflows.standard_frames import build_reference_club_overlay
        reference = np.zeros((19, 3))
        donor = np.arange(57, dtype=float).reshape(19, 3)
        points = np.zeros((19, 2))
        points[18] = [0, 40]
        detected = dict(points=points, valid=np.ones(19, bool), club_confidence=np.array([.9, .8]))
        overlay, details = build_reference_club_overlay(
            dict(reference=reference, aligned={'face_on': donor}), {'face_on': detected})
        self.assertTrue(details['applied'])
        self.assertEqual(details['landmark_order'], ['shaft', 'hosel'])
        np.testing.assert_array_equal(overlay[17:], donor[17:])
        np.testing.assert_array_equal(overlay[:17], reference[:17])

    def test_sequence_uses_golfpose_wrists_not_coco_indexes(self):
        model = GolferViTPose.__new__(GolferViTPose)
        xy = np.ones((22, 2)) * 30
        xy[13, 1], xy[16, 1] = 70, 90
        xy[9, 1], xy[10, 1] = 1, 2
        model.predict_frame = lambda frame, **kwargs: dict(points=xy.copy(), valid=np.ones(22, bool),
            box=np.array([0, 0, 100, 100]), keypoint_confidence=np.arange(22) / 22,
            head=None, grip=None)
        capture = Mock()
        capture.get.return_value = 25
        capture.read.side_effect = [(True, np.zeros((100, 100, 3))), (False, None)]
        with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture):
            sequence = model.detect_sequence('clip.mp4')
        self.assertEqual(sequence.wrist_y[0], 80)
        self.assertEqual(sequence.fps, 25)
        np.testing.assert_array_equal(sequence.points[0], xy)
        self.assertEqual(sequence.keypoint_confidence.shape, (1, 22))
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
