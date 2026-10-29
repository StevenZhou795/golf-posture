import unittest

import numpy as np

from golf_workflows.official_yolox import OfficialClubDetector, _prepare


class OfficialYOLOXTests(unittest.TestCase):
    def test_preprocess_matches_top_left_640_square_padding(self):
        frame = np.full((100, 200, 3), (1, 2, 3), dtype=np.uint8)
        tensor, scale = _prepare(frame)

        self.assertEqual(tuple(tensor.shape), (1, 3, 640, 640))
        self.assertEqual(scale, 3.2)
        np.testing.assert_array_equal(tensor[0, :, 0, 0].numpy(), [1, 2, 3])
        np.testing.assert_array_equal(tensor[0, :, 500, 0].numpy(), [114, 114, 114])

    def test_club_box_selects_highest_scoring_class_one_detection(self):
        detector = OfficialClubDetector.__new__(OfficialClubDetector)
        detector.metadata = {"club_crop_backend": "official-yolox-2cls"}
        detector.predict = lambda _frame: {
            "boxes": np.array([[0, 0, 5, 5], [10, 20, 30, 40], [11, 21, 31, 41]], dtype=float),
            "scores": np.array([.99, .4, .8]),
            "labels": np.array([0, 1, 1]),
            "inference_seconds": .01,
        }

        box, details = detector.club_box(np.zeros((50, 50, 3), dtype=np.uint8))

        np.testing.assert_array_equal(box, [11, 21, 31, 41])
        self.assertEqual(details["club_candidates"], 2)
        self.assertEqual(details["selected_club_score"], .8)


if __name__ == "__main__":
    unittest.main()
