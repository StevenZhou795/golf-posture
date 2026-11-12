import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from rtmw_workflow.swingtrace import associate, SwingTraceSequence, CLASS_NAMES
from rtmw_workflow.frame_evidence import evidence_for_frame


def detections(shift=0.):
    return [dict(class_id=c, box=(np.array(b)+[shift, 0, shift, 0]).tolist(), confidence=.9)
            for c, b in [(0, [12, 47, 88, 53]), (1, [86, 46, 94, 54]), (2, [6, 46, 14, 54])]]


class SwingTraceTests(unittest.TestCase):
    def test_class_two_is_grip_despite_label_three(self):
        self.assertEqual(CLASS_NAMES, {0: 'shaft', 1: 'head', 2: 'grip'})
        row = associate(detections(), [10, 50], [0, 0, 100, 100])
        self.assertTrue(row['pair_usable'])
        self.assertEqual(row['grip_anchor_source'], 'swingtrace_grip_box')
        np.testing.assert_allclose(row['proxy_xy'], [[50, 50], [90, 50]])

    def test_other_golfer_and_bright_line_do_not_win_on_score(self):
        other = detections(400)
        for r in other:
            r['confidence'] = .99
        row = associate(other+detections(), [10, 50], [0, 0, 100, 100])
        self.assertEqual(row['proxy_xy'][1], [90., 50.])
        self.assertFalse(associate(other, [10, 50], [0, 0, 100, 100])['pair_usable'])

    def test_empty_low_confidence_invalid_and_unanchored_stay_missing(self):
        rows = detections()
        for r in rows:
            r['confidence'] = .1
        rows += [dict(class_id=0, box=[0, 0, float('nan'), 10], confidence=.9)]
        for candidate, anchor in [(rows, [10, 50]), ([], [10, 50]), (detections(), None)]:
            row = associate(candidate, anchor, [0, 0, 100, 100])
            self.assertFalse(row['pair_usable'])
            self.assertIsNone(row['proxy_xy'][1])

    def test_missing_head_never_uses_shaft_box_corner(self):
        row = associate([r for r in detections() if r['class_id'] != 1], [10, 50], [0, 0, 100, 100])
        self.assertFalse(row['pair_usable'])
        self.assertIsNone(row['proxy_xy'][1])
        self.assertEqual(row['proxy_confidence'][1], 0.)

    def test_missing_grip_has_explicit_rtmw_wrist_fallback(self):
        row = associate(detections()[:2], [10, 50], [0, 0, 100, 100])
        self.assertTrue(row['pair_usable'])
        self.assertEqual(row['grip_anchor_source'], 'rtmw_wrists')

    def test_inconsistent_shaft_is_rejected(self):
        rows = detections()
        rows[0]['box'] = [5, 49, 15, 95]
        self.assertFalse(associate(rows, [10, 50], [0, 0, 100, 100])['pair_usable'])

    def test_sequence_contract_uses_nineteen_joints_and_one_detection_per_frame(self):
        from ultralytics.engine.results import Results
        points = np.full((1, 17, 3), 50., np.float32)
        points[:, :, 2] = .9
        result = Results(np.zeros((100, 100, 3), np.uint8), path='synthetic', names={0:'person'},
                         boxes=torch.tensor([[0., 0., 100., 100., .9, 0.]]), keypoints=torch.from_numpy(points))
        body = Mock(metadata={})
        body.predict.return_value = [result, result]
        body.predictor.dataset.fps = 30.
        club = Mock(confidence_threshold=.25)
        club.observe.return_value = (np.array([[50., 50.], [90., 50.]]), np.array([.9, .9]), {})
        swing = SwingTraceSequence(body, club).detect_sequence('not-a-video')
        self.assertEqual(swing.points.shape, (2, 19, 2))
        self.assertEqual(club.observe.call_count, 2)
        self.assertTrue(swing.valid[:, 17:19].all())

    def test_manual_ball_is_location_only_and_reads_no_video(self):
        from golf_workflows.golf_pose_3d import track_ball
        with patch('cv2.VideoCapture', side_effect=AssertionError('must not read video')):
            row = track_ball(Path('unused'), None, SimpleNamespace(size=(100, 100)), 0, 3, 'left', [90, 50])
        self.assertEqual(row['source'], 'manual_placement')
        self.assertIsNone(row['p1_frame'])
        self.assertIsNone(row['p7_clubhead_distance_px'])
        with self.assertRaises(ValueError):
            track_ball(Path('unused'), None, SimpleNamespace(size=(100, 100)), 0, 3, 'left', [100, 50])

    def test_head_proxies_and_ball_placement_feed_original_contact_rule(self):
        from golf_workflows.golf_pose_3d import contact_frame
        points = np.full((3, 19, 2), np.nan)
        valid = np.zeros((3, 19), bool)
        points[:, 18] = [[80, 50], [90, 50], [98, 50]]
        valid[:, 18] = True
        swing = SimpleNamespace(points=points, valid=valid)
        selected, source, _ = contact_frame(swing, [90, 50], (0, 2), 3.)
        self.assertEqual((selected, source), (1, 'clubhead_at_ball'))
        self.assertIsNone(contact_frame(swing, None, (0, 2), 3.)[0])

    def test_evidence_does_not_turn_proximity_into_contact_confirmation(self):
        row = associate(detections(), [10, 50], [0, 0, 100, 100])
        ev = evidence_for_frame('p7', 'wrist_return', row,
                               dict(ball=dict(p1_xy=[90, 50], source='manual_placement')))
        self.assertEqual(ev['head_centre_to_ball_px'], 0.)
        self.assertEqual(ev['selector_source'], 'wrist_return')
        self.assertIsNone(ev['impact_diagnostics'])
        self.assertEqual(ev['shaft_proxy_tilt_from_horizontal_deg'], 0.)

    def test_loader_never_constructs_vitpose_or_club_segmentation(self):
        from rtmw_workflow.cli import load_components
        from rtmw_workflow.swingtrace import CHECKPOINT
        with patch('golf_workflows.club_vitpose.load_club_pose_backend', side_effect=AssertionError('ViTPose forbidden')), \
             patch('golf_workflows.golf_pose_3d.load_lifter'), \
             patch('golf_workflows.official_yolox.OfficialClubDetector'), \
             patch('rtmw_workflow.runtime.load_model'), \
             patch('rtmw_workflow.backend.RTMWBodyPose'), \
             patch('rtmw_workflow.swingtrace.SwingTraceClub') as club, \
             patch('ultralytics.YOLO') as yolo, patch('ultralytics.YOLOE'):
            load_components(SimpleNamespace(checkpoint=Path('body.pth'), cpu_threads=2,
                pose_device='cpu', club_device='cpu', club_checkpoint=CHECKPOINT))
        club.assert_called_once_with(CHECKPOINT, 'cpu')
        self.assertEqual(yolo.call_count, 1)  # initial golfer ROI only

    def test_raw_pair_does_not_override_downstream_rejection_or_legacy_enum(self):
        from rtmw_workflow.frame_evidence import enrich_manifest
        observation = associate(detections(), [10, 50], [0, 0, 100, 100])
        observation['frame_index'] = 0
        manifest = dict(frames=[dict(position='p7', source='wrist_return', frame_index=0, club_evidence='none')])
        enrich_manifest(manifest, [observation], {})
        row = manifest['frames'][0]
        self.assertEqual(row['club_evidence'], 'none')
        self.assertTrue(row['evidence']['raw_pair_usable'])
        self.assertEqual(row['club_observation_kind'], 'partial_missing_or_rejected_boxes')

    def test_legacy_batch_cannot_mix_with_new_detector(self):
        from rtmw_workflow.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'clip.mp4').touch()
            out = root/'out'
            out.mkdir()
            (out/'batch_summary.json').write_text(json.dumps(dict(results=[])))
            with patch('rtmw_workflow.cli.load_components') as load:
                with self.assertRaises(SystemExit):
                    main(['batch', '--input-dir', str(root), '--output-dir', str(out), '--resume'])
            load.assert_not_called()
