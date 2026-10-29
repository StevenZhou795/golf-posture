"""Official GolfPose five-landmark ViTPose-H club inference.

Segmentation supplies only the top-down crop. The fine-tuned checkpoint supplies
the ordered landmarks consumed by GolfPose-3D: shaft, hosel, heel, toe_down,
toe_up.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from golf_workflows.vitpose_tracking import ROOT, crop_person, decode_heatmaps, load_checkpoint

CHECKPOINT = ROOT / "golfpose_checkpoints" / "golfpose_club_ViTPose_huge.pth"
LANDMARKS = ("shaft", "hosel", "heel", "toe_down", "toe_up")
CONFIDENCE = 0.10


class ClubViTPose:
    confidence_threshold = CONFIDENCE

    def __init__(self, checkpoint: Path = CHECKPOINT):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"GolfPose club checkpoint not found: {checkpoint}. Download "
                "golfpose_club_ViTPose_huge.pth from the official GolfPose model table."
            )
        self.model = load_checkpoint(checkpoint, keypoints=len(LANDMARKS))
        self.checkpoint = checkpoint
        self.metadata = {
            "club_pose_backend": "golfpose_club_ViTPose_huge",
            "club_pose_checkpoint": str(checkpoint.relative_to(ROOT)),
            "landmarks": list(LANDMARKS),
            "input_size_hw": [256, 192],
            "decoder": "UDPHeatmap + DARK refinement",
            "flip_test": True,
            "crop_source": "selected club bounding box",
            "strict_parameter_load": True,
        }

    @torch.inference_mode()
    def predict(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        image, center, scale, matrix = crop_person(frame, box)
        started = perf_counter()
        output = self.model(torch.cat((image, image.flip(-1))))
        # The GolfPose club dataset declares no left/right keypoint swaps.
        heatmap = (output[0] + output[1].flip(-1)) / 2
        xy, confidence = decode_heatmaps(heatmap.cpu().numpy(), center, scale)
        details = {
            "bbox_xyxy": np.asarray(box, dtype=float).tolist(),
            "crop_affine": matrix.tolist(),
            "confidence": confidence.tolist(),
            "landmark_xy": xy.tolist(),
            "inference_seconds": round(perf_counter() - started, 3),
        }
        return xy.astype(np.float64), confidence.astype(np.float32), details

    def predict_coarse_to_fine(self, frame: np.ndarray, person_box: np.ndarray):
        """Recover a club crop when localization missed it, then rerun all 5 points."""
        coarse_xy, coarse_confidence, coarse_details = self.predict(frame, person_box)
        vector = coarse_xy[1] - coarse_xy[0]
        length = float(np.linalg.norm(vector))
        if (coarse_confidence[:2] < self.confidence_threshold).any() or length < 12:
            coarse_details["crop_fallback"] = "golfer_box_only"
            return coarse_xy, coarse_confidence, coarse_details
        margin = max(12.0, 0.22 * length)
        low = np.minimum(coarse_xy[0], coarse_xy[1]) - margin
        high = np.maximum(coarse_xy[0], coarse_xy[1]) + margin
        height, width = frame.shape[:2]
        low = np.maximum(low, [0, 0])
        high = np.minimum(high, [width - 1, height - 1])
        shaft_box = np.concatenate((low, high))
        fine_xy, fine_confidence, fine_details = self.predict(frame, shaft_box)
        fine_details["crop_fallback"] = "golfer_box_then_supervised_shaft_hosel"
        fine_details["coarse_bbox_xyxy"] = np.asarray(person_box, dtype=float).tolist()
        fine_details["coarse_shaft_hosel_xy"] = coarse_xy[:2].tolist()
        fine_details["coarse_shaft_hosel_confidence"] = coarse_confidence[:2].tolist()
        return fine_xy, fine_confidence, fine_details


def load_club_pose_backend(backend: str):
    if backend == "segmentation":
        return None
    if backend == "vitpose-huge":
        return ClubViTPose()
    raise ValueError(f"Unknown club pose backend: {backend}")
