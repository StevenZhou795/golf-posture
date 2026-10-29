import unittest

import numpy as np

from golf_workflows.standard_frames import build_reference_club_overlay


class ReferenceClubOverlayTests(unittest.TestCase):
    def _fixture(self, valid=True):
        reference = np.arange(66, dtype=float).reshape(22, 3)
        donor = reference + 100.0
        points = np.arange(44, dtype=float).reshape(22, 2)
        points[17] = [10.0, 20.0]
        points[18] = [40.0, 20.0]
        flags = np.ones(22, dtype=bool)
        flags[20] = valid
        detections = {
            "face_on": {
                "points": points,
                "valid": flags,
                "club_confidence": np.array([.9, .8, .7, .6, .5]),
                "club_pose_details": {},
            }
        }
        fused = {"reference": reference, "aligned": {"face_on": donor}}
        return reference, donor, fused, detections

    def test_overlay_preserves_numerical_reference_and_channel_geometry(self):
        reference, donor, fused, detections = self._fixture()
        before = reference.copy()
        overlay, metadata = build_reference_club_overlay(fused, detections)

        np.testing.assert_array_equal(fused["reference"], before)
        np.testing.assert_array_equal(overlay[:17], before[:17])
        np.testing.assert_array_equal(overlay[17:22], donor[17:22])
        self.assertTrue(metadata["applied"])
        self.assertFalse(metadata["numerical_reference_modified"])
        self.assertEqual(metadata["landmark_order"], ["shaft", "hosel", "heel", "toe_down", "toe_up"])

    def test_overlay_rejects_incomplete_five_landmark_view(self):
        reference, _donor, fused, detections = self._fixture(valid=False)
        before = reference.copy()
        overlay, metadata = build_reference_club_overlay(fused, detections)

        self.assertIsNone(overlay)
        self.assertFalse(metadata["applied"])
        np.testing.assert_array_equal(fused["reference"], before)


if __name__ == "__main__":
    unittest.main()
