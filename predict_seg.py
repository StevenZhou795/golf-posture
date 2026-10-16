"""Segment golf clubs with the trained YOLO26n-seg weights.

Runs on images/ files that were not part of the training split. Copies with
the same bytes as a training image are skipped as well.

Usage:
    python predict_seg.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
TRAIN_DIR = ROOT / "datasets" / "golf-club-seg" / "images" / "train"
MODEL_PATH = ROOT / "runs" / "segment" / "golf_club_train" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "runs" / "segment"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_golf_club(name: str) -> bool:
    label = name.lower().replace("_", " ").replace("-", " ")
    return "golf club" in label


def held_out_images(images: list[Path], train_images: list[Path]) -> list[Path]:
    trained = {file_hash(path) for path in train_images}
    return [path for path in images if file_hash(path) not in trained]


def club_records(result, names: dict) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []

    clubs = []
    for index, class_id in enumerate(result.boxes.cls.cpu().tolist()):
        class_id = int(class_id)
        class_name = str(names[class_id])
        if not is_golf_club(class_name):
            continue

        polygon = []
        if result.masks is not None and index < len(result.masks.xy):
            polygon = result.masks.xy[index].tolist()

        clubs.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(result.boxes.conf[index].cpu()),
                "box_xyxy": result.boxes.xyxy[index].cpu().tolist(),
                "polygon": polygon,
            }
        )
    return clubs


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise SystemExit(f"Image folder not found: {IMAGE_DIR}")
    if not TRAIN_DIR.is_dir():
        raise SystemExit(f"Training images not found: {TRAIN_DIR}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Trained weights not found: {MODEL_PATH}")

    images = collect_images(IMAGE_DIR)
    train_images = collect_images(TRAIN_DIR)
    sources = held_out_images(images, train_images)
    if not sources:
        raise SystemExit(f"No images in {IMAGE_DIR} are outside the training split")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=[str(path) for path in sources],
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="heldout",
        exist_ok=True,
    )

    summary = []
    club_count = 0
    for result in results:
        clubs = club_records(result, model.names)
        club_count += len(clubs)
        summary.append(
            {
                "image": str(Path(result.path).relative_to(ROOT)),
                "clubs": clubs,
            }
        )

    output_dir = OUTPUT_DIR / "heldout"
    output_dir.mkdir(parents=True, exist_ok=True)
    clubs_path = output_dir / "clubs.json"
    clubs_path.write_text(
        json.dumps(
            {
                "model": str(MODEL_PATH.relative_to(ROOT)),
                "target": "golf club",
                "skipped_training_images": len(images) - len(sources),
                "images": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Skipped {len(images) - len(sources)} training images")
    print(f"Predicted {len(results)} images")
    print(f"Golf clubs: {club_count}")
    print(f"Annotated images: {output_dir}")
    print(f"Club masks: {clubs_path}")


if __name__ == "__main__":
    main()
