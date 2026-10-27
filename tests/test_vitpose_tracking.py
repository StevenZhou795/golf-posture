import unittest
from types import SimpleNamespace

import numpy as np
import torch

from golf_workflows.vitpose_tracking import ViTPoseTracking, crop_person, decode_heatmaps, select_golfer


class ViTPoseAdapterTests(unittest.TestCase):
    def test_udp_crop_maps_bbox_center_to_input_center(self):
        frame = np.zeros((300, 250, 3), dtype=np.uint8)
        image, center, scale, matrix = crop_person(frame, [50, 40, 150, 240])
        self.assertEqual(tuple(image.shape), (1, 3, 256, 192))
        np.testing.assert_allclose(center, [100, 140])
        np.testing.assert_allclose(scale, [187.5, 250])
        np.testing.assert_allclose(matrix @ [100, 140, 1], [95.5, 127.5], atol=1e-5)

    def test_udp_decoder_recovers_subpixel_gaussian_landmark(self):
        y, x = np.mgrid[:64, :48]
        heatmap = np.exp(-((x - 17.3) ** 2 + (y - 32.4) ** 2) / 8).astype(np.float32)
        heatmaps = np.repeat(heatmap[None], 17, axis=0)
        points, scores = decode_heatmaps(heatmaps, np.array([96, 128]), np.array([192, 256]))
        np.testing.assert_allclose(points[0], [17.3 / 47 * 192, 32.4 / 63 * 256], atol=.1)
        self.assertTrue((scores > .9).all())

    def test_video_association_keeps_prior_golfer(self):
        boxes = np.array([[10, 10, 60, 110], [100, 10, 190, 140]], dtype=float)
        self.assertEqual(select_golfer(boxes), 1)
        self.assertEqual(select_golfer(boxes, np.array([12, 10, 62, 110])), 0)
        self.assertEqual(select_golfer(np.empty((0, 4))), None)

    def test_no_golfer_does_not_invent_body_keypoints(self):
        model = ViTPoseTracking.__new__(ViTPoseTracking)
        model.model = lambda _: self.fail("No box should mean no pose inference")
        detected = SimpleNamespace(orig_img=np.zeros((100, 100, 3), np.uint8), boxes=None)
        result, box = model._pose(detected)
        self.assertIsNone(box)
        self.assertEqual(len(result.keypoints), 0)


if __name__ == "__main__":
    unittest.main()
