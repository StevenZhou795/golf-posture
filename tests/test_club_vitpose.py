import unittest

import numpy as np
import torch

from golf_workflows.club_vitpose import ClubViTPose, LANDMARKS


class ClubViTPoseTests(unittest.TestCase):
    def test_five_heatmap_channels_keep_official_joint_order(self):
        model = ClubViTPose.__new__(ClubViTPose)
        heatmaps = torch.zeros(2, 5, 64, 48)
        expected = []
        for index in range(5):
            x, y = 8 + index * 5, 10 + index * 7
            heatmaps[:, index, y, x] = 1
            # Mirrored inference is flipped back before decoding.
            heatmaps[1, index].zero_()
            heatmaps[1, index, y, 47 - x] = 1
            # GetBBoxCenterScale expands this full-frame box by 1.25.
            expected.append([x / 47 * 240 - 24, y / 63 * 320 - 32])
        model.model = lambda _image: heatmaps
        points, confidence, details = model.predict(
            np.zeros((256, 192, 3), dtype=np.uint8), np.array([0, 0, 192, 256])
        )
        self.assertEqual(LANDMARKS, ("shaft", "hosel", "heel", "toe_down", "toe_up"))
        np.testing.assert_allclose(points, expected, atol=0.2)
        self.assertEqual(confidence.shape, (5,))
        self.assertEqual(len(details["confidence"]), 5)

    def test_golfer_fallback_reruns_on_supervised_shaft_segment(self):
        model = ClubViTPose.__new__(ClubViTPose)
        calls = []

        def predict(_frame, box):
            calls.append(np.asarray(box, dtype=float))
            if len(calls) == 1:
                xy = np.array([[100, 80], [110, 180], [0, 0], [0, 0], [0, 0]], dtype=float)
                return xy, np.array([.8, .7, 0, 0, 0], dtype=float), {}
            return np.ones((5, 2)), np.ones(5), {"bbox_xyxy": calls[-1].tolist()}

        model.predict = predict
        _xy, _confidence, details = model.predict_coarse_to_fine(
            np.zeros((240, 320, 3), dtype=np.uint8), np.array([20, 10, 250, 230])
        )
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(calls[1], [77.89, 57.89, 132.11, 202.11], atol=.02)
        self.assertEqual(details["crop_fallback"], "golfer_box_then_supervised_shaft_hosel")


if __name__ == "__main__":
    unittest.main()
