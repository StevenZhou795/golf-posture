import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rtmw_workflow.backend import RTMWBodyPose, extract_prediction
from rtmw_workflow.cli import main, inputs, run_video
from rtmw_workflow.device import ClubDeviceModule
from rtmw_workflow.report import selection_source


def prediction(score=.9, shift=0):
    return SimpleNamespace(transformed_keypoints=np.full((1, 133, 2), -999.),
        keypoint_scores=np.full((1, 133), score), keypoints=np.full((1, 133, 2), 20.+shift))


class AdapterTests(unittest.TestCase):
    def test_2d_estimator_pixels_get_only_the_outer_crop_offset(self):
        sample = prediction()
        xy, confidence = extract_prediction(sample, (100, 50))
        np.testing.assert_array_equal(xy, np.tile([120., 70.], (133, 1)))
        np.testing.assert_array_equal(sample.keypoints, np.full((1, 133, 2), 20.))

    def test_3d_predictions_cannot_be_misread_as_2d_pixels(self):
        sample = prediction()
        sample.keypoints = np.zeros((1, 133, 3))
        with self.assertRaisesRegex(ValueError, '2D prediction'):
            extract_prediction(sample)

    def backend(self):
        detector = Mock(metadata={})
        detector.predict.return_value = dict(boxes=np.array([[100., 50., 200., 150.]]),
            scores=np.array([.9]), labels=np.array([0]))
        return RTMWBodyPose(Mock(), detector, Path('test.pth'))

    def test_coco_subset_and_crop_recovery_save_winner(self):
        body = self.backend()
        body.set_initial_golfer_box([80., 30., 220., 170.])
        responses = [[SimpleNamespace(pred_instances=prediction(.9))],
                     [SimpleNamespace(pred_instances=prediction(.1, 40))]]
        # Force the first candidate to need a retry while still winning by score.
        with patch('rtmw_workflow.official.inference', side_effect=responses), \
             patch.object(body, '_target_pose', side_effect=lambda boxes, points: (boxes, points)), \
             patch.object(body, '_quality', side_effect=[(False, 15.), (False, 2.)]):
            result, _ = body.predict_frame(np.zeros((200, 300, 3), np.uint8))
        self.assertEqual(result.keypoints.data.shape, (1, 17, 3))
        np.testing.assert_allclose(body.current[0][0], [100., 50.])
        np.testing.assert_allclose(result.keypoints.data[0, :, 2], .9)
        self.assertEqual(body.current[0].shape, (133, 2))

    def test_unanchored_detector_miss_stays_missing(self):
        body = self.backend()
        body.detector.predict.return_value = dict(boxes=np.empty((0, 4)), scores=np.array([]), labels=np.array([]))
        with patch('rtmw_workflow.official.inference', return_value=[SimpleNamespace(pred_instances=prediction())]):
            result, _ = body.predict_frame(np.zeros((200, 300, 3), np.uint8))
        self.assertIsNone(body.current)
        self.assertEqual(len(result.keypoints), 0)

    def test_requested_threads_restored_after_roi_predictor(self):
        body = self.backend()
        body.cpu_threads = 6
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30.
        capture.read.return_value = (False, None)
        with patch('rtmw_workflow.backend.cv2.VideoCapture', return_value=capture), \
             patch('rtmw_workflow.backend.torch.set_num_threads') as set_threads:
            self.assertEqual(body.predict('empty.mp4'), [])
        set_threads.assert_called_once_with(6)
        capture.release.assert_called_once()

    def test_rejected_identity_is_not_saved_as_wholebody_prediction(self):
        body = self.backend()
        body.last_body_box = np.array([100., 50., 200., 150.])
        body.last_body_keypoints = np.tile([150., 90., .9], (17, 1))
        body.lock_anchor = np.array([150., 90., 100.])
        with patch('rtmw_workflow.official.inference', return_value=[SimpleNamespace(pred_instances=prediction())]), \
             patch.object(body, '_target_pose', side_effect=lambda boxes, points: (boxes[:0], points[:0])):
            result, _ = body.predict_frame(np.zeros((200, 300, 3), np.uint8))
        self.assertIsNone(body.current)
        self.assertEqual(len(result.keypoints), 0)
        self.assertTrue(result.detector_details['target_locked'])
        np.testing.assert_array_equal(body.last_body_box, [100., 50., 200., 150.])

    def test_nan_confidence_and_missing_frames(self):
        p = prediction()
        p.keypoints[0, 5] = np.nan
        p.keypoint_scores[0, 9] = np.nan
        xy, scores = extract_prediction(p)
        self.assertEqual(scores[5], 0)
        self.assertEqual(scores[9], 0)
        body = self.backend()
        body.records = [dict(prediction=None, time_ms=0., timestamp_fallback=False, detector={})]
        with tempfile.TemporaryDirectory() as tmp:
            body.save_raw(Path(tmp))
            with np.load(Path(tmp)/'wholebody.npz') as data:
                self.assertNotIn('camera_3d', data.files)
                self.assertTrue(np.isnan(data['image_2d']).all())
                self.assertFalse(data['confidence'].any())


class PipelineTests(unittest.TestCase):
    def test_cached_body_replay_preserves_records_without_inference(self):
        body = AdapterTests().backend()
        expected = [object(), object()]
        body.replay_results = expected
        body.records = [dict(frame_index=0), dict(frame_index=1)]
        with patch('rtmw_workflow.backend.cv2.VideoCapture') as capture:
            self.assertEqual(list(body.predict('clip.mp4', stream=True)), expected)
        capture.assert_not_called()
        self.assertEqual(len(body.records), 2)

    def test_incomplete_swing_is_not_exported_as_ten_valid_phases(self):
        from rtmw_workflow.report import export
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'clip_3d.json').write_text(json.dumps(dict(phase_detection=dict(
                fallback='no_corroborated_wrist_rise_and_return'))))
            with patch('golf_workflows.cli.export_identified_frames') as exporter:
                with self.assertRaisesRegex(ValueError, 'Untrusted P1-P10'):
                    export(folder, Path('clip.mp4'))
            exporter.assert_not_called()

    def test_negative_decoder_start_time_uses_monotonic_fallback(self):
        body = AdapterTests().backend()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [30., -33.333, 0.]
        capture.read.side_effect = [(True, np.zeros((2, 2, 3), np.uint8)),
                                    (True, np.zeros((2, 2, 3), np.uint8)), (False, None)]
        body.current = None
        with patch('rtmw_workflow.backend.cv2.VideoCapture', return_value=capture), \
             patch('rtmw_workflow.backend.torch.set_num_threads'), \
             patch.object(body, 'predict_frame', return_value=(SimpleNamespace(detector_details={}), None)):
            list(body.predict('clip.mp4', stream=True))
        np.testing.assert_allclose([r['time_ms'] for r in body.records], [0., 1000/30])
        self.assertTrue(all(r['timestamp_fallback'] for r in body.records))

    def test_anchor_evidence_is_exported_without_inventing_confidence(self):
        summary = dict(phase_detection=dict(impact_source='wrist_return'),
                       checkpoint_selection=dict(selector='joint_dp', position_sources=dict(p2='lead_arm_progress')))
        self.assertEqual(selection_source('p7', summary), 'wrist_return')
        self.assertEqual(selection_source('p2', summary), 'lead_arm_progress')
        self.assertEqual(selection_source('p3', summary), 'lead_arm_horizontal_rising')

    def test_device_wrapper_retains_float32_batch_and_cpu_output(self):
        model = torch.nn.Conv2d(3, 5, 1).eval()
        images = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4)
        expected = model(images).detach()
        wrapper = ClubDeviceModule(model, 'cpu')
        with torch.inference_mode():
            output = wrapper(images)
        self.assertEqual(output.device.type, 'cpu')
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.shape[0], 2)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_root_videos_only_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'a.mp4').touch()
            (root/'b.mp4').touch()
            (root/'image.jpg').touch()
            (root/'nested').mkdir()
            (root/'nested/c.mp4').touch()
            args = SimpleNamespace(command='batch', input_dir=root)
            self.assertEqual(len(inputs(args)), 2)
            args.priority = ['b.mp4']
            self.assertEqual([p.name for p in inputs(args)], ['b.mp4', 'a.mp4'])
            with patch('rtmw_workflow.cli.load_components', return_value={}), \
                 patch('rtmw_workflow.scripts.audit_results.audit', return_value={'results': []}), \
                 patch('rtmw_workflow.cli.run_video', side_effect=[SystemExit('no golfer hips'), {'frames': []}]):
                code = main(['batch', '--input-dir', str(root), '--output-dir', str(root/'out')])
            report = json.loads((root/'out/batch_summary.json').read_text())
            self.assertEqual(code, 2)
            self.assertEqual((report['success'], report['failed']), (1, 1))

    def test_resume_preserves_interrupted_attempt_and_skips_completed(self):
        from rtmw_workflow.runtime import CHECKPOINT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a.mp4', 'b.mp4'):
                (root/name).touch()
            out = root/'out'
            (out/'b').mkdir(parents=True)
            (out/'b/partial.txt').write_text('preserve')
            (out/'batch_summary.json').write_text(json.dumps(dict(results=[dict(
                video=str((root/'a.mp4').resolve()), status='success')], input_count=2,
                club_backend='swingtrace-yolo11m', pose_backend='rtmw-x-2d',
                pose_checkpoint=str(CHECKPOINT.resolve()))))
            with patch('rtmw_workflow.cli.load_components', return_value={}), \
                 patch('rtmw_workflow.scripts.audit_results.audit', return_value={'results': []}), \
                 patch('rtmw_workflow.cli.run_video', return_value={'frames': []}) as run:
                code = main(['batch', '--input-dir', str(root), '--output-dir', str(out), '--resume'])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0].name, 'b.mp4')
            self.assertEqual(next(out.glob('b_interrupted_*/partial.txt')).read_text(), 'preserve')

    def test_original_process_called_without_rule_overrides(self):
        body = Mock(metadata={})
        components = dict(body=body, lifter=object(), club=Mock(metadata={}),
                          detector=Mock(metadata={}), ball=Mock(overrides={}), model=Mock())
        args = SimpleNamespace(golfer_box=[1, 2, 30, 40], handedness='auto', view='auto')
        with tempfile.TemporaryDirectory() as tmp, \
             patch('golf_workflows.golf_pose_3d.process') as shared, \
             patch('rtmw_workflow.report.export', return_value={'frames': []}), \
             patch('rtmw_workflow.report.render_wholebody'):
            folder = Path(tmp)
            run_video(Path('005.mp4'), folder, components, args)
        shared.assert_called_once()
        self.assertIs(shared.call_args.args[1].body, body)
        self.assertTrue(shared.call_args.args[1].joint_pose)
        self.assertEqual(shared.call_args.kwargs['club_angle_measurements'][0], {})
        self.assertNotIn('club_crop_model', shared.call_args.kwargs)
        self.assertNotIn('club_mask_model', shared.call_args.kwargs)
        self.assertIs(shared.call_args.kwargs['frame_overlay'], components['club'].draw)
        self.assertIs(shared.call_args.args[3], components['lifter'])
        self.assertTrue(shared.call_args.kwargs['club_refinement'])
        self.assertFalse(shared.call_args.kwargs['allow_skeleton_gaps'])
        self.assertNotIn('reviewed_positions', shared.call_args.kwargs)
        body.save_raw.assert_called_once()


if __name__ == '__main__':
    unittest.main()
