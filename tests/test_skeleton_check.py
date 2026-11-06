import tempfile
import unittest
from pathlib import Path

import numpy as np

from golf_workflows.golf_pose_3d import SkeletonQualityError, Swing2D, check_skeleton, skeleton_lost_frames


def _swing(valid, fps=30.0):
    n = len(valid)
    points = np.zeros((n, 19, 2))
    return Swing2D(frames=n, fps=fps, size=(100, 100), points=points, valid=valid, filled=points,
                   boxes=np.zeros((n, 4)), samples=[], wrist_y=np.zeros(n))


class SkeletonCheckTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def test_short_top_occlusion_passes(self):
        valid = np.ones((60, 19), dtype=bool)
        valid[20:27, 11:17] = False  # arms behind the head for 7 frames (0.23 s)
        report = check_skeleton(_swing(valid), 0, 59, self.folder, "swing window")
        self.assertTrue(report["passed"])
        self.assertEqual(report["lost_ranges"], [[20, 26]])
        self.assertTrue((self.folder / "skeleton_diagnostics.json").is_file())

    def test_lost_golfer_aborts_with_ranges(self):
        valid = np.ones((120, 19), dtype=bool)
        valid[80:, :17] = False  # crop slid off the golfer for the rest of the clip
        with self.assertRaises(SkeletonQualityError) as caught:
            check_skeleton(_swing(valid), 10, 119, self.folder, "swing window")
        self.assertIn("80-119", str(caught.exception))
        self.assertIn("--allow-skeleton-gaps", str(caught.exception))

    def test_missing_both_wrists_counts_as_lost(self):
        valid = np.ones((3, 19), dtype=bool)
        valid[1, [13, 16]] = False
        self.assertEqual(skeleton_lost_frames(valid).tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
