import unittest
from types import SimpleNamespace

import numpy as np

from blazepose_workflow import landmarks, smoothing
from blazepose_workflow.inference import monotonic_ms, select_golfer
from blazepose_workflow.positions import POSITIONS, SwingCoverageError, select_positions
from blazepose_workflow.tests.synthetic import swing


class LandmarkTests(unittest.TestCase):
    def test_face_landmarks_are_dropped(self):
        self.assertEqual(len(landmarks.NAMES), 23)
        self.assertEqual(landmarks.SOURCE_INDEX[:3], [0, 11, 12])
        self.assertFalse(any(index in landmarks.SOURCE_INDEX for index in range(1, 11)))
        pose = [SimpleNamespace(x=i, y=-i, z=0.5, visibility=0.9, presence=1.0) for i in range(33)]
        self.assertEqual(landmarks.subset(pose)[1].tolist(), [11, -11, 0.5, 0.9, 1.0])

    def test_lead_side_follows_handedness(self):
        self.assertEqual(landmarks.lead_trail("right"), ("left", "right"))
        self.assertEqual(landmarks.lead_trail("left"), ("right", "left"))
        with self.assertRaises(ValueError):
            landmarks.lead_trail("auto")

    def test_hands_ignore_missing_points(self):
        points = np.zeros((1, 23, 2))
        points[0, landmarks.J["left_wrist"]] = np.nan
        points[0, landmarks.J["right_wrist"]] = [6, 6]
        self.assertTrue(np.allclose(landmarks.hands(points)[0], [1.2, 1.2]))


class SmoothingTests(unittest.TestCase):
    def test_short_gaps_fill_in_time_long_gaps_stay_missing(self):
        times = np.array([0, 33, 100, 133, 166, 600, 633], float)
        series = np.array([0, np.nan, 10, 11, np.nan, np.nan, 20], float).reshape(-1, 1, 1)
        filled = smoothing.fill_gaps(series, times)[:, 0, 0]
        self.assertAlmostEqual(filled[1], 3.3)
        self.assertTrue(np.isnan(filled[4:6]).all())

    def test_smoothing_keeps_a_straight_line(self):
        times = np.arange(30) * 1000 / 30
        line = np.linspace(0, 1, 30).reshape(-1, 1, 1)
        self.assertTrue(np.allclose(smoothing.smooth(line, times), line))

    def test_bone_lengths_are_held_to_the_median(self):
        _, world, _ = swing()
        scaled = world * np.linspace(.8, 1.2, len(world))[:, None, None]
        lengths = smoothing.bone_lengths(smoothing.normalize_bones(scaled))
        # Cross bones (the shoulder line) follow from the two shoulder chains.
        held = [i for i, bone in enumerate(landmarks.BONES)
                if bone != (landmarks.J["left_shoulder"], landmarks.J["right_shoulder"])]
        self.assertTrue(np.allclose(lengths[:, held], lengths[0, held], atol=1e-9))

    def test_visibility_gate(self):
        gated = smoothing.gate(np.ones((1, 2, 2)), np.array([[.9, .2]]))
        self.assertTrue(np.isnan(gated[0, 1]).all() and np.isfinite(gated[0, 0]).all())


class InferenceTests(unittest.TestCase):
    def test_timestamps_strictly_increase(self):
        self.assertEqual(monotonic_ms([0, 33.3, 33.3, 66.7, 66.0, 100]), [0, 33, 34, 67, 68, 100])

    def test_central_golfer_beats_edge_bystander_then_is_followed(self):
        def person(x, height):
            pose = np.zeros((23, 5))
            pose[:, 3] = 1
            pose[:, 0] = x
            pose[:, 1] = np.linspace(500 - height, 500, 23)
            return pose
        golfer, guard = person(400, 380), person(40, 400)
        self.assertEqual(select_golfer([guard, golfer], None, None, (720, 1280)), 1)
        anchor = (golfer[landmarks.J["left_hip"], :2] + golfer[landmarks.J["right_hip"], :2]) / 2
        self.assertEqual(select_golfer([guard, person(420, 380)], anchor, 380.0, (720, 1280)), 1)
        self.assertIsNone(select_golfer([guard], anchor, 380.0, (720, 1280)))


class PositionTests(unittest.TestCase):
    def check(self, handedness):
        points2d, world, times = swing(handedness)
        result = select_positions(points2d, world, times)
        frames = [result["positions"][key] for key in POSITIONS]
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual((result["view"], result["handedness"]), ("face_on", handedness))
        self.assertEqual(result["positions"]["p4"], 40)
        self.assertIn(result["positions"]["p1"], range(13, 17))
        self.assertIn(result["positions"]["p7"], range(54, 57))
        self.assertTrue(15 < result["positions"]["p2"] < result["positions"]["p3"] < 40)
        self.assertTrue(40 < result["positions"]["p5"] < result["positions"]["p6"] < result["positions"]["p7"])
        self.assertTrue(result["positions"]["p7"] < result["positions"]["p8"] < result["positions"]["p9"] < 75)
        self.assertIn(result["positions"]["p10"], range(74, 78))
        p3 = result["details"]["p3"]
        self.assertTrue(times[result["positions"]["p3"] - 1] <= p3["time_ms"] <= times[result["positions"]["p3"]])
        return result

    def test_right_handed_swing(self):
        self.check("right")

    def test_left_handed_swing(self):
        self.check("left")

    def test_clip_starting_at_the_top_is_rejected(self):
        points2d, world, times = swing()
        with self.assertRaises(SwingCoverageError) as caught:
            select_positions(points2d[40:], world[40:], times[40:] - times[40])
        self.assertIn("starts at or after the top", str(caught.exception))

    def test_clip_ending_before_the_follow_through_is_rejected(self):
        points2d, world, times = swing()
        with self.assertRaises(SwingCoverageError) as caught:
            select_positions(points2d[:60], world[:60], times[:60])
        self.assertIn("ends before the follow-through", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
