import unittest

import numpy as np

from golf_workflows.golfer_roi import roi_from_person_boxes


class GolferRoiTests(unittest.TestCase):
    def test_union_of_agreeing_boxes_is_expanded_for_the_club(self):
        boxes = [[100, 200, 200, 400], [110, 190, 210, 400], [90, 200, 200, 410], [500, 0, 590, 100]]
        roi, kept = roi_from_person_boxes(boxes, (600, 800))
        self.assertEqual(len(kept), 3)
        height = 410 - 190
        np.testing.assert_allclose(roi, [90 - .15 * height, 190 - .25 * height, 210 + .15 * height, 410 + .05 * height])

    def test_roi_is_clipped_to_the_frame(self):
        roi, _ = roi_from_person_boxes([[0, 10, 50, 200]] * 3, (60, 205))
        np.testing.assert_allclose(roi, [0, 0, 59, 204])

    def test_too_few_person_frames_gives_no_roi(self):
        roi, _ = roi_from_person_boxes([[0, 0, 10, 10]] * 2, (100, 100))
        self.assertIsNone(roi)


if __name__ == '__main__':
    unittest.main()
