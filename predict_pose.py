"""Run YOLO26n-pose on every image under images/.

Usage:
    pip install ultralytics
    python predict_pose.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = ROOT / "runs" / "pose"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Index order from https://docs.ultralytics.com/tasks/pose
COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def person_record(result, index: int) -> dict:
    keypoints = result.keypoints
    xy = keypoints.xy[index].cpu().tolist()
    confidence = None
    if keypoints.conf is not None:
        confidence = keypoints.conf[index].cpu().tolist()

    points = []
    for keypoint_index, (x, y) in enumerate(xy):
        name = (
            COCO_KEYPOINTS[keypoint_index]
            if keypoint_index < len(COCO_KEYPOINTS)
            else f"kpt_{keypoint_index}"
        )
        point = {"name": name, "x": x, "y": y}
        if confidence is not None:
            point["confidence"] = confidence[keypoint_index]
        points.append(point)

    person = {"keypoints": points}
    boxes = result.boxes
    if boxes is not None and index < len(boxes):
        person["box_xyxy"] = boxes.xyxy[index].cpu().tolist()
        person["confidence"] = float(boxes.conf[index].cpu())
    return person


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")

    images = collect_images(IMAGE_DIR)
    if not images:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    model = YOLO("yolo26n-pose.pt")
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="predict",
        exist_ok=True,
    )

    summary = []
    for result in results:
        count = 0 if result.keypoints is None else len(result.keypoints)
        summary.append(
            {
                "image": str(Path(result.path).relative_to(ROOT)),
                "people": [person_record(result, index) for index in range(count)],
            }
        )

    keypoints_path = OUTPUT_DIR / "predict" / "keypoints.json"
    keypoints_path.parent.mkdir(parents=True, exist_ok=True)
    keypoints_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Predicted {len(results)} images")
    print(f"Annotated images: {keypoints_path.parent}")
    print(f"Keypoints: {keypoints_path}")


if __name__ == "__main__":
    main()
