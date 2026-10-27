import json
import tempfile
import unittest
from pathlib import Path

from golf_workflows.golf_pose_3d import load_reviewed_positions


class ReviewedPositionTests(unittest.TestCase):
    def selection(self):
        return {"source":"swing.mp4", "frame_index_base":0,
                "frames":[{"position":f"P{i}","frame_index":i*10} for i in range(1,11)]}

    def load(self, data):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder)/"selection.json"
            path.write_text(json.dumps(data))
            return load_reviewed_positions(path, Path("swing.mp4"))

    def test_saved_reviewed_frames_load_in_order(self):
        self.assertEqual(list(self.load(self.selection()).values()), list(range(10,101,10)))

    def test_wrong_video_is_rejected(self):
        data = self.selection(); data["source"] = "other.mp4"
        with self.assertRaisesRegex(ValueError,"source"):
            self.load(data)

    def test_repeated_or_reversed_frame_indices_are_rejected(self):
        data = self.selection(); data["frames"][4]["frame_index"] = 10
        with self.assertRaisesRegex(ValueError,"increasing"):
            self.load(data)

    def test_missing_position_is_rejected(self):
        data = self.selection(); data["frames"][-1]["position"] = "P9"
        with self.assertRaisesRegex(ValueError,"P1 through P10"):
            self.load(data)


if __name__ == "__main__":
    unittest.main()
