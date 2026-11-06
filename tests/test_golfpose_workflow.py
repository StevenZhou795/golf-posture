import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from golfpose_workflow import cli as joint
from golf_workflows import cli as shared
from golf_workflows.golf_pose_3d import joint_names


class JointWorkflowTests(unittest.TestCase):
    def setUp(self):
        scratch = shared.ROOT / 'runs' / '.workflow_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name).resolve()
        if not self.root.is_relative_to(scratch.resolve()):
            raise ValueError('Unexpected test cleanup path')
        self.addCleanup(self.temp.cleanup)
        self.joint_root = self.root / 'joint'
        self.old_root = self.root / 'original'
        self.addCleanup(patch.stopall)
        patch.object(joint, 'RUNS', self.joint_root).start()
        patch.object(shared, 'RUNS', self.old_root).start()
        self.old_root.mkdir()
        shared.publish('video', self.root / 'original-video')
        self.old_pointer = (self.old_root / 'latest_video.json').read_bytes()

    def test_fresh_commands_pin_single_native_model_without_changing_original_default(self):
        for option, handler in (('--reference', 'references'), ('--video', 'video')):
            argv = [option] + (['clip.mp4'] if option == '--video' else [])
            with patch.object(shared, handler) as call:
                joint.main(argv)
                args = call.call_args.args[0]
                self.assertEqual(args.pose_backend, joint.BACKEND)
                self.assertEqual(args.lifter, '17+2')
                self.assertIsNone(args.from_run)
                self.assertEqual(args.runs_root, self.joint_root)
        with patch.object(shared, 'references') as call:
            shared.main(['references'])
            self.assertEqual(call.call_args.args[0].pose_backend, 'yolo26n-pose')

    def fake_video(self, command):
        out = Path(command[command.index('--run-dir') + 1])
        poses = np.repeat(self.pose[None], 35, axis=0)
        confidence = np.full((35, 19), .8)
        np.savez(out / 'clip_3d.npz', joints=np.array(joint_names(19)),
                 keypoints_3d=poses, keypoints_2d_confidence=confidence,
                 club_landmark_confidence=confidence[:, 17:])
        shared.write_json(out / 'clip_3d.json', dict(source='clip.mp4', lead_side='left',
            pose_backend={'pose_backend': joint.BACKEND},
            positions=dict(zip(shared.POSITIONS, self.indices))))
        for key in shared.POSITIONS:
            cv2.imwrite(str(out / f'{key.upper()}.jpg'), np.zeros((8, 8, 3), np.uint8))

    def test_video_exports_original_ten_indices_and_scoring_matches_existing_rules(self):
        self.pose = np.random.default_rng(88).normal(size=(19, 3))
        self.indices = list(range(2, 32, 3))
        clip = self.root / 'clip.mp4'
        clip.write_bytes(b'test-input')
        with patch('golf_workflows.golf_pose_3d.main', side_effect=self.fake_video) as process, \
             patch('golf_workflows.standard_frames.save_still'):
            joint.main(['--video', str(clip), '--golfer-box', '10', '20', '100', '200'])
        command = process.call_args.args[0]
        self.assertIn('--club-refinement', command)
        self.assertEqual(command[command.index('--pose-backend') + 1], joint.BACKEND)
        roi_index = command.index('--golfer-box')
        self.assertEqual(command[roi_index+1:roi_index+5], ['10.0', '20.0', '100.0', '200.0'])
        video = shared.latest('video', self.joint_root)
        with np.load(video / 'identified_frames.npz') as data:
            self.assertEqual(data['keypoints_3d'].shape, (10, 19, 3))
            self.assertEqual(data['keypoints_2d_confidence'].shape, (10, 19))
            np.testing.assert_array_equal(data['source_frame_indices'], self.indices)
        self.assertEqual(shared.read_json(video / 'workflow.json')['club_landmarks'], ['shaft', 'hosel'])
        refs = self.joint_root / 'references'
        for key in shared.POSITIONS:
            folder = refs / f'{key.upper()}-reference'
            folder.mkdir(parents=True)
            np.savez(folder / 'joints.npz', reference_3d=self.pose)
            cv2.imwrite(str(folder / 'reference_3d.jpg'), np.zeros((8, 8, 3), np.uint8))
        shared.publish('references', refs, self.joint_root)
        with patch('golf_workflows.compare_standard_frames.save_still'), \
             patch('golf_workflows.compare_standard_frames.side_by_side'):
            joint.main(['--score'])
            score = shared.read_json(shared.latest('score', self.joint_root) / 'score.json')
            shared.score(argparse.Namespace(run=video, references=refs,
                output_dir=self.root / 'original-score'))
            baseline = shared.read_json(self.root / 'original-score' / 'score.json')
        self.assertEqual(score['pose_score'], 100)
        self.assertEqual(score['positions'], baseline['positions'])
        self.assertEqual(score['criteria'], baseline['criteria'])
        self.assertEqual(score['coverage']['parameters_scored'], 140)
        self.assertEqual(list(score['frames'].values()), self.indices)
        self.assertEqual(score['references'], str(refs))
        self.assertEqual((self.old_root / 'latest_video.json').read_bytes(), self.old_pointer)

    def test_score_does_not_fall_back_to_original_workflow_pointers(self):
        with self.assertRaises(SystemExit) as caught:
            joint.main(['--score'])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse((self.joint_root / 'latest_score.json').exists())

    def test_failed_ten_frame_validation_does_not_publish_video(self):
        self.pose = np.ones((19, 3))
        self.indices = [2] * 10
        clip = self.root / 'clip.mp4'
        clip.write_bytes(b'test-input')
        with patch('golf_workflows.golf_pose_3d.main', side_effect=self.fake_video), \
             self.assertRaises(SystemExit):
            joint.main(['--video', str(clip)])
        self.assertFalse((self.joint_root / 'latest_video.json').exists())
        self.assertEqual((self.old_root / 'latest_video.json').read_bytes(), self.old_pointer)


if __name__ == '__main__':
    unittest.main()
