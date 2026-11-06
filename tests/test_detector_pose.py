import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from golf_workflows.detector_pose import YOLOXBodyPose
from golf_workflows.golfer_vitpose import GolferViTPose
from golf_workflows.golfer_vitpose import validate_golfer_box
from golf_workflows.official_yolox import OfficialClubDetector, _decode


class DetectorPoseTests(unittest.TestCase):
    def test_yolo_pose_crop_restores_original_coordinates_and_reuses_detector_for_club(self):
        detector = OfficialClubDetector.__new__(OfficialClubDetector)
        detector.metadata = {'club_crop_backend': 'official-yolox-2cls'}
        detector.predict = Mock(return_value=dict(boxes=np.array([[100., 50., 200., 150.], [130., 120., 250., 160.]]),
            scores=np.array([.9, .8]), labels=np.array([0, 1]), inference_seconds=.01))
        pose = Mock(metadata={'pose_backend': 'yolo26n-pose'})
        def infer(frame, **kwargs):
            keypoints = torch.tensor([[[10., 20., .9]] * 17])
            return [Results(frame, path='', names={0: 'person'}, boxes=torch.tensor([[0., 0., 100., 100., .9, 0.]]),
                            keypoints=keypoints)]
        pose.predict.side_effect = infer
        frame = np.zeros((200, 300, 3), np.uint8)
        result, box = YOLOXBodyPose(pose, detector).predict_frame(frame)
        np.testing.assert_allclose(result.keypoints.data[0, 0, :2], [105, 65])
        self.assertIs(result.orig_img, frame)
        self.assertEqual(pose.predict.call_args.args[0].shape, (110, 110, 3))
        club, _ = detector.club_box(frame, result=result.yolox_detection, person_box=box)
        np.testing.assert_array_equal(club, [130, 120, 250, 160])
        detector.predict.assert_called_once()

    def test_club_crop_ignores_high_scoring_banner_far_from_golfer(self):
        detector = OfficialClubDetector.__new__(OfficialClubDetector)
        detector.metadata = {}
        result = dict(boxes=np.array([[100., 0., 250., 30.], [150., 600., 400., 850.]]),
            scores=np.array([.99, .7]), labels=np.array([1, 1]), inference_seconds=.01)
        club, details = detector.club_box(np.zeros((1000, 600, 3)), result=result,
            person_box=np.array([100., 450., 300., 900.]), anchor=np.array([220., 650.]))
        np.testing.assert_array_equal(club, [150, 600, 400, 850])
        self.assertEqual(details['associated_club_candidates'], 1)

    def test_decode_supports_one_and_two_class_checkpoint_heads(self):
        for count in (1, 2):
            cls = torch.full((1, count, 1, 1), -8.)
            cls[0, count-1, 0, 0] = 8.
            boxes, scores, labels = _decode([(cls, torch.zeros(1, 4, 1, 1), torch.full((1, 1, 1, 1), 8.))], .25, .65, 1.)
            self.assertEqual(labels.tolist(), [count-1])
            np.testing.assert_allclose(boxes, [[-4, -4, 4, 4]])
            self.assertGreater(scores[0], .99)

    def joint_model(self, box):
        model = GolferViTPose.__new__(GolferViTPose)
        model.detector = Mock()
        model.detector.golfer_box.return_value = (box, {})
        model.model = Mock(return_value=torch.zeros((2, 22, 64, 48)))
        return model

    def test_joint_pose_uses_one_class_box_with_original_frame_coordinate_decode(self):
        box = np.array([100., 200., 250., 400.])
        model = self.joint_model(box)
        frame = np.zeros((600, 800, 3), np.uint8)
        xy = np.full((22, 2), [150., 250.], dtype=np.float32)
        with patch('golf_workflows.golfer_vitpose.crop_person',
                   return_value=(torch.zeros(1, 3, 256, 192), None, None, np.eye(2, 3))) as crop, \
             patch('golf_workflows.golfer_vitpose.decode_heatmaps', return_value=(xy, np.ones(22))):
            result = model.predict_frame(frame)
        np.testing.assert_array_equal(crop.call_args.args[1], box)
        np.testing.assert_array_equal(result['points'], xy)
        self.assertEqual(result['pose_backend_details']['crop_source'], 'official-yolox-1cls')
        self.assertTrue(result['valid'].all())

    def test_video_detector_miss_does_not_run_pose_on_full_frame(self):
        model = self.joint_model(None)
        result = model.predict_frame(np.zeros((600, 800, 3), np.uint8), allow_panel_fallback=False)
        model.model.assert_not_called()
        self.assertFalse(result['valid'].any())
        self.assertTrue(np.isnan(result['points']).all())
        self.assertEqual(result['detector_details']['crop_source'], 'detector-miss')

    def test_reference_detector_miss_is_explicit_panel_fallback(self):
        model = self.joint_model(None)
        frame = np.zeros((256, 192, 3), np.uint8)
        with patch('golf_workflows.golfer_vitpose.decode_heatmaps',
                   return_value=(np.full((22, 2), 30.), np.ones(22))):
            result = model.predict_frame(frame)
        model.model.assert_called_once()
        self.assertEqual(result['pose_backend_details']['crop_source'], 'reference-panel-fallback')
        self.assertIsNone(result['detector_box'])

    def test_video_gap_reuses_last_detector_roi_without_fabricating_a_detection(self):
        model = self.joint_model(None)
        frame = np.zeros((600, 800, 3), np.uint8)
        previous = np.array([100., 200., 250., 400.])
        with patch('golf_workflows.golfer_vitpose.decode_heatmaps',
                   return_value=(np.full((22, 2), [150., 250.]), np.ones(22))):
            result = model.predict_frame(frame, previous_box=previous, allow_panel_fallback=False)
        model.model.assert_called_once()
        self.assertEqual(result['detector_details']['crop_source'], 'tracked-detector-roi')
        np.testing.assert_array_equal(result['detector_details']['pose_input_bbox_xyxy'], previous)
        self.assertIsNone(result['detector_box'])

    def test_explicit_roi_bootstraps_sequence_when_detector_never_finds_golfer(self):
        model = self.joint_model(None)
        model.metadata = {}
        model.set_initial_golfer_box([10, 20, 90, 95])
        capture = Mock()
        capture.get.return_value = 30
        capture.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8)), (False, None)]
        with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture), \
             patch('golf_workflows.golfer_vitpose.decode_heatmaps',
                   return_value=(np.full((22, 2), 50.), np.ones(22))):
            swing = model.detect_sequence('clip.mp4')
        self.assertTrue(swing.valid[0, 0])
        self.assertEqual(swing.detector_details[0]['crop_source'], 'manual-golfer-roi')
        np.testing.assert_array_equal(swing.detector_details[0]['pose_input_bbox_xyxy'], [10, 20, 90, 95])
        self.assertEqual(model.metadata['initial_roi_source'], 'manual')

    def test_automatic_roi_bootstraps_sequence_and_is_labelled(self):
        model = self.joint_model(None)
        model.metadata = {}
        model.set_initial_golfer_box([10, 20, 90, 95], 'auto-yolo26n', {'agreeing_frames': 12})
        capture = Mock()
        capture.get.return_value = 30
        capture.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8)), (False, None)]
        with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture), \
             patch('golf_workflows.golfer_vitpose.decode_heatmaps',
                   return_value=(np.full((22, 2), 50.), np.ones(22))):
            swing = model.detect_sequence('clip.mp4')
        self.assertEqual(swing.detector_details[0]['crop_source'], 'auto-golfer-roi')
        self.assertEqual(model.metadata['initial_roi_details'], {'agreeing_frames': 12})

    def test_body_crop_uses_golfer_roi_when_yolox_misses_person(self):
        detector = OfficialClubDetector.__new__(OfficialClubDetector)
        detector.metadata = {}
        detector.predict = Mock(return_value=dict(boxes=np.empty((0, 4)), scores=np.empty(0),
            labels=np.empty(0, dtype=np.int64), inference_seconds=.01))
        pose = Mock(metadata={'pose_backend': 'yolo26n-pose'})
        pose.predict.side_effect = lambda frame, **kwargs: [Results(frame, path='', names={0: 'person'},
            boxes=torch.tensor([[0., 0., 10., 10., .9, 0.]]), keypoints=torch.tensor([[[1., 2., .9]] * 17]))]
        body = YOLOXBodyPose(pose, detector)
        frame = np.zeros((200, 300, 3), np.uint8)
        result, _ = body.predict_frame(frame)
        self.assertEqual(result.detector_details['body_crop_source'], 'full-frame-fallback')
        body.set_initial_golfer_box([50, 20, 150, 180], 'auto-yolo26n')
        result, _ = body.predict_frame(frame)
        self.assertEqual(result.detector_details['body_crop_source'], 'auto-golfer-roi')
        self.assertEqual(pose.predict.call_args.args[0].shape, (160, 100, 3))
        np.testing.assert_allclose(result.keypoints.data[0, 0, :2], [51, 22])

    def test_detector_initialization_failure_saves_diagnostics_with_actionable_error(self):
        model = self.joint_model(None)
        model.metadata = {}
        capture = Mock()
        capture.get.return_value = 30
        capture.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8)), (False, None)]
        scratch = Path(__file__).resolve().parents[1] / 'runs' / '.workflow_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            if not Path(folder).resolve().is_relative_to(scratch.resolve()):
                raise ValueError('Unexpected test cleanup path')
            model.diagnostics_dir = folder
            with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture), \
                 self.assertRaisesRegex(ValueError, 'ViTPose inference was skipped.*--golfer-box'):
                model.detect_sequence('clip.mp4')
            report = json.loads((Path(folder) / 'pose_inference_diagnostics.json').read_text())
            self.assertEqual(report['pose_inference_frames'], 0)
            self.assertEqual(report['decoded_frames'], 1)
            with np.load(Path(folder) / 'pose_inference_diagnostics.npz') as data:
                self.assertFalse(data['keypoints_2d_valid'].any())
        model.model.assert_not_called()

    def test_root_confidence_failure_is_distinguished_from_detector_miss(self):
        model = self.joint_model(None)
        valid = np.ones(22, bool)
        valid[0] = False
        confidence = np.ones(22)
        confidence[0] = .01
        model.predict_frame = Mock(return_value=dict(points=np.full((22, 2), 50.), valid=valid,
            box=np.array([10, 20, 90, 95]), keypoint_confidence=confidence, head=None, grip=None))
        capture = Mock()
        capture.get.return_value = 30
        capture.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8)), (False, None)]
        with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture), \
             self.assertRaisesRegex(ValueError, 'ViTPose ran on 1/1 ROI crops.*highest root confidence 0.010'):
            model.detect_sequence('clip.mp4')

    def test_manual_roi_rejects_invalid_geometry(self):
        for box in ([100, 20, 10, 90], [-1, 0, 20, 40], [1, 2, float('nan'), 40], [1, 2, 3]):
            with self.assertRaises(ValueError):
                validate_golfer_box(box)

    def test_nonfinite_pose_output_is_not_reported_as_a_missing_detector_roi(self):
        model = self.joint_model(None)
        model.predict_frame = Mock(return_value=dict(points=np.full((22, 2), np.nan), valid=np.zeros(22, bool),
            box=np.full(4, np.nan), keypoint_confidence=np.zeros(22), head=None, grip=None,
            detector_details={'crop_source': 'manual-golfer-roi'}))
        capture = Mock()
        capture.get.return_value = 30
        capture.read.side_effect = [(True, np.zeros((100, 100, 3), np.uint8)), (False, None)]
        with patch('golf_workflows.golfer_vitpose.cv2.VideoCapture', return_value=capture), \
             self.assertRaisesRegex(ValueError, 'ViTPose ran on 1/1 ROI crops'):
            model.detect_sequence('clip.mp4')


if __name__ == '__main__':
    unittest.main()
