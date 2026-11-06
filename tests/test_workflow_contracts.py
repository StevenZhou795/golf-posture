import json
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from golf_workflows.cli import (POSITIONS, export_identified_frames,
                                load_identified_frames, new_output, validate_reference_set, video)
from golf_workflows.golf_pose_3d import GOLFPOSE_JOINTS
from golf_workflows.compare_standard_frames import compare


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        workspace = Path(__file__).resolve().parents[1]
        scratch = workspace / 'runs' / '.workflow_tests'
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        # Verify the recursive temporary cleanup stays inside this workspace.
        if not self.root.resolve().is_relative_to(scratch.resolve()):
            raise ValueError('Unexpected temporary test directory')
        self.addCleanup(self.temp.cleanup)
        self.pose = np.random.default_rng(12).normal(size=(19,3))

    def video_fixture(self):
        folder = self.root / 'video'
        folder.mkdir()
        poses = np.repeat(self.pose[None], 35, axis=0)
        np.savez(folder/'swing_3d.npz', keypoints_3d=poses, joints=np.array(GOLFPOSE_JOINTS))
        indices = list(range(2,32,3))
        (folder/'swing_3d.json').write_text(json.dumps(dict(source='swing.mp4',lead_side='left',
            positions=dict(zip(POSITIONS,indices)))), encoding='utf-8')
        for key in POSITIONS:
            cv2.imwrite(str(folder/f'{key.upper()}.jpg'),np.zeros((8,8,3),np.uint8))
        with patch('golf_workflows.standard_frames.save_still'):
            export_identified_frames(folder)
        return folder, indices

    def reference_fixture(self):
        refs = self.root/'references'
        refs.mkdir()
        for key in POSITIONS:
            folder=refs/f'{key.upper()}-reference'
            folder.mkdir()
            np.savez(folder/'joints.npz',reference_3d=self.pose)
            cv2.imwrite(str(folder/'reference_3d.jpg'),np.zeros((8,8,3),np.uint8))
        return refs

    def test_frame_export_retains_22_joint_confidence_and_shaft_fallback_evidence(self):
        from golf_workflows.golf_pose_3d import joint_names
        folder, indices = self.video_fixture()
        points = np.ones((35, 22, 2)) * 10
        points[:, 18] = [80, 10]
        valid = np.ones((35, 22), bool)
        valid[indices[0], 18:] = False
        points[indices[0], 18:] = np.nan
        confidence = np.full((35, 22), .9)
        np.savez(folder/'swing_3d.npz', keypoints_3d=np.ones((35, 22, 3)),
            joints=np.array(joint_names(22)), keypoints_2d=points,
            keypoints_2d_valid=valid, keypoints_2d_confidence=confidence,
            club_landmark_confidence=confidence[:, 17:])
        with patch('golf_workflows.standard_frames.save_still'):
            manifest = export_identified_frames(folder)
        with np.load(folder/'identified_frames.npz') as data:
            np.testing.assert_array_equal(data['keypoints_2d_confidence'], confidence[indices])
            self.assertFalse(data['keypoints_2d_valid'][0, 18:].any())
            self.assertEqual(data['club_evidence'][0], 'shaft_fallback')
        self.assertEqual(manifest['frames'][0]['club_evidence'], 'shaft_fallback')

    def test_selected_bundle_scores_supplied_references_and_retains_source_indices(self):
        folder,indices=self.video_fixture()
        refs=self.reference_fixture()
        self.assertEqual(len(validate_reference_set(refs)),10)
        selected=load_identified_frames(folder)
        with patch('golf_workflows.compare_standard_frames.save_still'), patch('golf_workflows.compare_standard_frames.side_by_side'):
            result=compare(folder,refs,self.root/'scores',selected)
        self.assertEqual(result['pose_score'],100)
        self.assertEqual(list(result['frames'].values()),indices)
        self.assertEqual(sum(len(v['parameters']) for v in result['positions'].values()),140)
        self.assertTrue((self.root/'scores'/'score.json').is_file())
        self.assertFalse((folder/'standard_compare').exists())

    def test_manifest_coordinate_mismatch_rejected(self):
        folder,_=self.video_fixture()
        path=folder/'identified_frames.json'
        manifest=json.loads(path.read_text())
        manifest['frames'][0]['frame_index']=99
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'disagree'):
            load_identified_frames(folder)

    def test_incomplete_reference_set_rejected(self):
        refs=self.reference_fixture()
        (refs/'P10-reference'/'joints.npz').unlink()
        with self.assertRaisesRegex(ValueError,'missing'):
            validate_reference_set(refs)

    def test_existing_default_output_is_preserved_and_explicit_overwrite_rejected(self):
        folder=self.root/'output'
        folder.mkdir()
        (folder/'existing.txt').write_text('preserve')
        alternate=new_output(None,folder)
        self.assertNotEqual(alternate,folder)
        self.assertEqual((folder/'existing.txt').read_text(),'preserve')
        with self.assertRaises(ValueError):
            new_output(folder,folder)

    def test_fresh_video_dispatch_uses_exact_output_directory_and_refinement(self):
        input_video=self.root/'input.mp4'
        input_video.write_bytes(b'placeholder')
        args=argparse.Namespace(video=input_video,from_run=None,positions_json=None,
              output_dir=self.root/'chosen-output',pose_backend='vitpose-huge',lifter='17+5',
              pose_cpu_threads=8,handedness='right',view='down_the_line',
              no_club_refinement=False,club_marker_length_m=None)
        with patch('golf_workflows.golf_pose_3d.main') as process, patch('golf_workflows.cli.export_identified_frames',return_value={}), patch('golf_workflows.cli.publish'):
            video(args)
        command=process.call_args.args[0]
        self.assertEqual(command[command.index('--run-dir')+1],str(args.output_dir.resolve()))
        self.assertIn('--club-refinement',command)
        self.assertIn('vitpose-huge',command)


if __name__=='__main__':
    unittest.main()
