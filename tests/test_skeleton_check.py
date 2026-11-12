import tempfile
import unittest
from pathlib import Path

import numpy as np

from golf_workflows.golf_pose_3d import (SkeletonQualityError, Swing2D, check_skeleton, skeleton_lost_frames,
                                         trailing_absence, trim_swing)


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

    def test_outro_without_golfer_is_trimmed(self):
        # 009: golfer scored ~0.93 until frame 419, then an end card with low-score blips.
        swing = _swing(np.ones((120, 19), dtype=bool))
        scores = [.93] * 90 + [None] * 10 + [.3] * 3 + [None] * 17
        swing.detector_details = [dict(frame_index=i, body=dict(selected_person_score=s)) for i, s in enumerate(scores)]
        self.assertEqual(trailing_absence(swing), 90)
        trimmed = trim_swing(swing, 90)
        self.assertEqual((trimmed.frames, len(trimmed.points), len(trimmed.detector_details)), (90, 90, 90))
        swing.detector_details = [dict(frame_index=i, selected_golfer_score=.9) for i in range(115)] + [None] * 5
        self.assertIsNone(trailing_absence(swing))  # 5 frames is within the gap limit


if __name__ == "__main__":
    unittest.main()
