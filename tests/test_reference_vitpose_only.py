"""Exercise reference club routing without loading neural-network packages."""

import ast
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np


class ReferenceViTPoseOnlyTests(unittest.TestCase):
    def setUp(self):
        # Isolate the routing function so this contract can also be checked in
        # environments without PyTorch, OpenCV, or Ultralytics installed.
        path = Path(__file__).resolve().parents[1] / 'golf_workflows' / 'standard_frames.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'detect_view')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
                            names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
        self.body_result = Mock()
        self.body_result.keypoints.xy = [Mock()]
        self.body_result.keypoints.xy[0].cpu.return_value.numpy.return_value = np.zeros((17, 2))
        self.body_result.keypoints.conf = None
        # Use a real container for the boxes, with the tensor API used by inference.
        class Boxes(list):
            pass
        self.body_result.boxes = Boxes([0])
        box_tensor = Mock()
        box_tensor.cpu.return_value.numpy.return_value = np.array([10, 20, 180, 240])
        self.body_result.boxes.xyxy = [box_tensor]
        self.namespace = dict(np=np, BODY_JOINTS=17, GOLFPOSE_JOINTS=list(range(19)),
                              golfer_index=lambda result: 0,
                              body_anchor=lambda result, person: None,
                              coco_to_golfpose=lambda xy, conf: (xy, np.ones(17, dtype=bool)))
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), self.namespace)
        self.pose = Mock()
        self.pose.predict.return_value = [self.body_result]
        self.club = Mock(confidence_threshold=.1)
        self.landmarks = np.arange(10, dtype=float).reshape(5, 2) + 30
        self.confidence = np.array([.9, .8, .7, .6, .5])
        self.club.predict_coarse_to_fine.return_value = (self.landmarks, self.confidence, {})

    def test_club_coordinates_come_only_from_vitpose_without_segmentation(self):
        detected = self.namespace['detect_view'](
            np.zeros((256, 192, 3)), self.pose, None, self.club)
        np.testing.assert_array_equal(detected['points'][17:22], self.landmarks)
        self.assertTrue(detected['valid'][17:22].all())
        np.testing.assert_array_equal(
            self.club.predict_coarse_to_fine.call_args.args[1], [10, 20, 180, 240])
        self.club.predict.assert_not_called()
        self.assertEqual(detected['club_crop_details']['club_crop_backend'], 'golfer-box')

    def test_missing_golfer_does_not_invoke_club_model_or_segmentation(self):
        self.namespace['golfer_index'] = lambda result: None
        detected = self.namespace['detect_view'](
            np.zeros((256, 192, 3)), self.pose, None, self.club)
        self.assertFalse(detected['found'])
        self.assertFalse(detected['valid'][17:22].any())
        self.club.predict_coarse_to_fine.assert_not_called()
        self.club.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
