import unittest
import numpy as np

from golf_workflows.standard_frames import body_view_score, fuse_views


class ReferenceBodySelectionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(17)
        poses = {view: rng.normal(size=(22, 3)) for view in ('face_on', 'down_the_line', 'rear')}
        detections = {view: dict(keypoint_confidence=np.full(22, score), valid=np.ones(22, bool))
                      for view, score in [('face_on', .9), ('down_the_line', .7), ('rear', 1)]}
        return poses, detections

    def test_higher_scoring_face_on_body_is_preserved_without_component_replacement(self):
        poses, detections = self.fixture()
        result = fuse_views(poses, detections)
        self.assertEqual(result['body_selection']['source_view'], 'face_on')
        np.testing.assert_array_equal(result['reference'], result['aligned']['face_on'])
        self.assertIsNone(result['side_axis'])
        self.assertEqual(result['kept'], ['face_on'])

    def test_higher_scoring_dtl_is_copied_as_a_whole_aligned_pose(self):
        poses, detections = self.fixture()
        detections['down_the_line']['keypoint_confidence'][:] = .95
        result = fuse_views(poses, detections)
        self.assertEqual(result['body_selection']['source_view'], 'down_the_line')
        np.testing.assert_array_equal(result['reference'], result['aligned']['down_the_line'])

    def test_club_confidence_does_not_affect_body_selection(self):
        poses, detections = self.fixture()
        detections['face_on']['keypoint_confidence'][17:] = 0
        result = fuse_views(poses, detections)
        self.assertEqual(result['body_selection']['source_view'], 'face_on')
        self.assertAlmostEqual(result['body_selection']['scores']['face_on'], .9)

    def test_invalid_joints_contribute_zero_and_ties_prefer_face_on(self):
        poses, detections = self.fixture()
        detections['face_on']['valid'][:10] = False
        self.assertAlmostEqual(body_view_score(detections['face_on']), .9 * 7 / 17)
        self.assertEqual(fuse_views(poses, detections)['body_selection']['source_view'], 'down_the_line')
        detections['face_on'] = detections['down_the_line']
        self.assertEqual(fuse_views(poses, detections)['body_selection']['source_view'], 'face_on')

    def test_rear_is_not_a_body_source(self):
        poses, detections = self.fixture()
        result = fuse_views({'rear': poses['rear']}, detections)
        self.assertIsNone(result['reference'])


if __name__ == '__main__':
    unittest.main()
