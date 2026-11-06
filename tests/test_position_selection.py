import unittest

import numpy as np

from golf_workflows.position_selection import infer_handedness, select_positions


def follow_through(n=50, impact=20, step=7.0):
    """Straight arms hanging (-80 deg) at impact, then rising `step` degrees a frame; shaft vertical."""
    poses = np.zeros((n, 19, 3))
    poses[:, 0] = [0, 0.5, 0]
    poses[:, 8] = [0, 0, 0]
    poses[:, 10] = [0, -0.3, 0]
    poses[:, 11] = [0.2, 0, 0]
    poses[:, 14] = [-0.2, 0, 0]
    elevation = np.full(n, -80.0)
    elevation[impact + 1:] = -80.0 + step * np.arange(1, n - impact)
    elevation[:10] = -80.0 + 9 * np.arange(10)
    elevation[10:impact - 2] = np.linspace(10.0, -80.0, impact - 12)
    for frame in range(n):
        direction = np.array([np.cos(np.radians(elevation[frame])), -np.sin(np.radians(elevation[frame])), 0])
        for shoulder, elbow, wrist in ((11, 12, 13), (14, 15, 16)):
            poses[frame, elbow] = poses[frame, shoulder] + .3 * direction
            poses[frame, wrist] = poses[frame, shoulder] + .6 * direction
    poses[:, 17] = poses[:, 13]
    poses[:, 18] = poses[:, 13] + [0, 1, 0]
    return poses, elevation


class P8Tests(unittest.TestCase):
    def test_unseen_shaft_uses_lead_arm_at_release_height(self):
        poses, elevation = follow_through()
        observed = np.zeros(len(poses), dtype=bool)
        positions, details = select_positions(poses, 0, 10, 20, club_observed=observed)
        # Lead arm 45% of the way from hanging (-80) to parallel: -44 deg, frame 25.1.
        self.assertEqual(positions['p8'], 25)
        self.assertEqual(positions['p9'], 31)
        self.assertEqual(details['sources']['p8'], 'lead_arm_progress')
        self.assertLess(positions['p8'], positions['p9'])

    def test_parallel_shaft_while_arms_still_hang_is_ignored(self):
        poses, _ = follow_through()
        # A "parallel" shaft while the arms still hang (misdetection), then the real one.
        for frame in (21, 22, 23):
            poses[frame, 18] = poses[frame, 17] + [1, 0, 0]
        for frame in (26, 27, 28):
            poses[frame, 18] = poses[frame, 17] + [1, 0.1 * (frame - 27), 0]
        observed = np.ones(len(poses), dtype=bool)
        positions, details = select_positions(poses, 0, 10, 20, club_observed=observed)
        self.assertEqual(positions['p8'], 27)
        self.assertEqual(details['sources']['p8'], 'observed_shaft')

    def test_face_on_p9_uses_observed_lead_arm(self):
        poses, _ = follow_through()
        image = poses.copy()
        # The lifted arm lags the picture by a frame after impact.
        poses[21:] = poses[20:-1]
        lifted, _ = select_positions(poses, 0, 10, 20)
        observed, details = select_positions(poses, 0, 10, 20, image_points=image)
        self.assertEqual(observed['p9'], lifted['p9'] - 1)
        self.assertEqual(details['p9_signal'], 'image_lead_arm')


class HandednessTests(unittest.TestCase):
    def test_face_on_uses_backswing_top_not_higher_finish(self):
        n = 40
        points = np.zeros((n, 19, 2))
        valid = np.ones((n, 19), dtype=bool)
        points[:, 0] = [0, 100]
        points[:, 8] = [0, 0]
        x = np.interp(np.arange(n), [0, 5, 12, 20, 30, 39], [0, 0, -60, 0, 80, 80])
        y = np.interp(np.arange(n), [0, 5, 12, 20, 30, 39], [80, 80, -100, 80, -160, 40])
        points[:, 13] = points[:, 16] = np.column_stack([x, y])
        handedness, evidence = infer_handedness(points, valid, 'face_on')
        self.assertEqual(handedness, 'right')
        self.assertEqual(evidence['top_frame'], 12)
        self.assertEqual(evidence['finish_frame'], 30)


if __name__ == '__main__':
    unittest.main()
