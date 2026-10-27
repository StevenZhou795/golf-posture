import unittest
from types import SimpleNamespace

import numpy as np
import torch

from golf_workflows.golf_pose_3d import fill_gaps, joint_names, lift_sequence, pad_keypoints, swap_sides


class LifterJointLayoutTests(unittest.TestCase):
    def test_extra_inputs_are_missing_not_measurements(self):
        original = np.arange(38, dtype=float).reshape(1, 19, 2)
        valid = np.ones((1, 19), dtype=bool)
        points, flags = pad_keypoints(original, valid, 22)
        np.testing.assert_array_equal(points[:, :19], original)
        self.assertTrue(np.isnan(points[:, 19:]).all())
        self.assertFalse(flags[:, 19:].any())
        filled = fill_gaps(points, flags)
        np.testing.assert_array_equal(filled[:, :19], original)
        np.testing.assert_array_equal(filled[:, 19:], np.repeat(original[:, 18:19], 3, axis=1))
        self.assertTrue(np.isnan(points[:, 19:]).all())

    def test_actual_extra_observation_survives_gap_filling(self):
        points = np.arange(44, dtype=float).reshape(1, 22, 2)
        valid = np.ones((1, 22), dtype=bool)
        valid[:, 20:] = False
        points[:, 20:] = np.nan
        filled = fill_gaps(points, valid)
        np.testing.assert_array_equal(filled[:, 19], points[:, 19])
        np.testing.assert_array_equal(filled[:, 20:], np.repeat(points[:, 18:19], 2, axis=1))

    def test_20_point_input_is_rejected_for_22_point_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "expects 22"):
            lift_sequence(SimpleNamespace(Spatial_pos_embed=torch.zeros(1, 22, 512)), np.zeros((1, 20, 2)))
        with self.assertRaises(ValueError):
            joint_names(20)

    def test_body_mirror_leaves_club_identity_and_is_reversible(self):
        points = torch.arange(44).reshape(1, 1, 22, 2)
        mirrored = swap_sides(points)
        self.assertTrue(torch.equal(mirrored[:, :, 17:], points[:, :, 17:]))
        self.assertTrue(torch.equal(swap_sides(mirrored), points))


if __name__ == "__main__":
    unittest.main()
