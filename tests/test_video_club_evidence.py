import unittest
from types import SimpleNamespace
import numpy as np

from golf_workflows.golf_pose_3d import club_frames, video_club_evidence


class VideoClubEvidenceTests(unittest.TestCase):
    def fixture(self):
        points = np.ones((3, 22, 2)) * 10
        points[:, 18] = [80, 10]
        valid = np.ones((3, 22), bool)
        valid[1, 18:] = False
        points[1, 18:] = np.nan
        valid[2, 17] = False
        return SimpleNamespace(frames=3, points=points, valid=valid,
            keypoint_confidence=np.full((3, 22), .9), club_confidence=None)

    def test_complete_and_shaft_only_frames_are_displayable_without_validating_missing_head(self):
        swing = self.fixture()
        self.assertEqual(video_club_evidence(swing).tolist(), ['complete', 'shaft_fallback', 'none'])
        np.testing.assert_array_equal(club_frames(swing), [0, 1])
        self.assertFalse(swing.valid[1, 18:].any())

    def test_low_confidence_shaft_is_not_displayable(self):
        swing = self.fixture()
        swing.keypoint_confidence[1, 17] = .05
        self.assertEqual(video_club_evidence(swing)[1], 'none')


if __name__ == '__main__':
    unittest.main()
