import unittest

import numpy as np

from golf_workflows.club_tracking import apply_track, track_club


def _club(angles_deg, length=200.0, hands=(300.0, 300.0), confidence=0.9):
    n = len(angles_deg)
    points = np.full((n, 19, 2), np.nan)
    valid = np.zeros((n, 19), dtype=bool)
    hand = np.tile(np.asarray(hands, float), (n, 1))
    for frame, angle in enumerate(np.radians(angles_deg)):
        grip = hand[frame] + 20.0 * np.array([np.cos(angle), np.sin(angle)])
        points[frame, 17] = grip
        points[frame, 18] = grip + length * np.array([np.cos(angle), np.sin(angle)])
        valid[frame, 17:19] = True
    return points, valid, np.full(n, confidence), hand


class ClubTrackingTests(unittest.TestCase):
    def test_confident_fast_downswing_is_not_reversed(self):
        # 70 degrees per frame near impact: the reversed direction can sit closer
        # to a constant-speed prediction, but the grip is at the hands.
        angles = [-120, -118, -115, -100, -60, 10, 80, 150, 200, 230]
        points, valid, confidence, hands = _club(angles)
        track = track_club(points, valid, confidence, hands, fps=28.0)
        self.assertFalse(track.swapped.any())
        self.assertFalse(track.rejected.any())
        np.testing.assert_allclose(track.points[:, 1], points[:, 18])

    def test_reversed_endpoints_are_fixed_by_geometry(self):
        points, valid, confidence, hands = _club([90] * 8)
        points[4, 17], points[4, 18] = points[4, 18].copy(), points[4, 17].copy()
        track = track_club(points, valid, confidence, hands, fps=30.0)
        self.assertEqual(np.flatnonzero(track.swapped).tolist(), [4])
        fixed, _ = apply_track(points, valid, track)
        np.testing.assert_allclose(fixed[4, 18], points[3, 18])

    def test_low_confidence_outlier_is_rejected(self):
        points, valid, confidence, hands = _club([90] * 10)
        angle = np.radians(0)
        points[5, 18] = points[5, 17] + 200.0 * np.array([np.cos(angle), np.sin(angle)])
        confidence[5] = 0.2
        track = track_club(points, valid, confidence, hands, fps=30.0)
        self.assertEqual(np.flatnonzero(track.rejected).tolist(), [5])
        self.assertLess(abs(np.degrees(track.theta[5]) - 90.0), 5.0)
        _, cleaned = apply_track(points, valid, track)
        self.assertFalse(cleaned[5, 18])


if __name__ == "__main__":
    unittest.main()
