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

    def test_incomplete_club_is_shown_when_no_complete_view_exists(self):
        reference, donor, fused, detections = self._fixture(valid=False)
        before = reference.copy()
        overlay, metadata = build_reference_club_overlay(fused, detections)

        self.assertTrue(metadata["applied"])
        self.assertTrue(metadata['shaft_only_fallback'])
        self.assertEqual(metadata['inferred_club_landmarks'], ['toe_down'])
        np.testing.assert_array_equal(overlay[17:22], donor[17:22])
        np.testing.assert_array_equal(fused["reference"], before)

    def test_confident_shaft_alone_is_sufficient_when_all_views_incomplete(self):
        _, donor, fused, detections = self._fixture()
        detected = detections['face_on']
        detected['valid'][18:22] = False
        detected['points'][18:22] = np.nan
        detected['club_confidence'][1:] = np.nan
        overlay, metadata = build_reference_club_overlay(fused, detections)
        self.assertTrue(metadata['applied'])
        self.assertEqual(metadata['observed_club_landmarks'], ['shaft'])
        self.assertEqual(metadata['inferred_club_landmarks'], ['hosel', 'heel', 'toe_down', 'toe_up'])
        self.assertIsNone(metadata['source_shaft_length_px'])
        self.assertIsNone(metadata['source_shaft_direction_2d'])
        np.testing.assert_array_equal(overlay[17:22], donor[17:22])

    def test_missing_or_low_confidence_shaft_cannot_qualify_for_fallback(self):
        _, _, fused, detections = self._fixture(valid=False)
        detections['face_on']['club_confidence'][0] = .05
        self.assertIsNone(build_reference_club_overlay(fused, detections)[0])
        detections['face_on']['club_confidence'][0] = .9
        detections['face_on']['valid'][17] = False
        self.assertIsNone(build_reference_club_overlay(fused, detections)[0])

    def test_incomplete_views_are_ranked_by_shaft_confidence(self):
        _, donor, fused, detections = self._fixture(valid=False)
        detections['down_the_line'] = {**detections['face_on'],
            'club_confidence': np.array([.95, .2, .2, .2, .2])}
        fused['aligned']['down_the_line'] = donor + 10
        overlay, metadata = build_reference_club_overlay(fused, detections)
        self.assertEqual(metadata['source_view'], 'down_the_line')
        np.testing.assert_array_equal(overlay[17:22], (donor + 10)[17:22])

    def test_higher_score_dtl_wins_but_rear_is_never_a_donor(self):
        reference, donor, fused, detections = self._fixture()
        detections['down_the_line'] = {**detections['face_on'], 'club_confidence': np.full(5, .85)}
        detections['rear'] = {**detections['face_on'], 'club_confidence': np.ones(5)}
        fused['aligned']['down_the_line'] = donor + 10
        fused['aligned']['rear'] = donor + 1000
        overlay, metadata = build_reference_club_overlay(fused, detections)
        self.assertEqual(metadata['source_view'], 'down_the_line')
        np.testing.assert_array_equal(overlay[17:22], (donor + 10)[17:22])
        np.testing.assert_array_equal(overlay[:17], reference[:17])

    def test_invalid_head_markers_disqualify_even_a_high_scoring_view(self):
        _, donor, fused, detections = self._fixture()
        flags = detections['face_on']['valid'].copy()
        flags[21] = False
        detections['down_the_line'] = {**detections['face_on'],
            'club_confidence': np.ones(5), 'valid': flags}
        fused['aligned']['down_the_line'] = donor + 10
        _, metadata = build_reference_club_overlay(fused, detections)
        self.assertEqual(metadata['source_view'], 'face_on')

    def test_rear_only_does_not_produce_overlay(self):
        _, donor, fused, detections = self._fixture()
        detections['rear'] = detections.pop('face_on')
        fused['aligned'] = {'rear': donor}
        overlay, metadata = build_reference_club_overlay(fused, detections)
        self.assertIsNone(overlay)
        self.assertFalse(metadata['applied'])


if __name__ == "__main__":
    unittest.main()
