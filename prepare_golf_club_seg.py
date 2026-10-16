"""Build a YOLO instance-segmentation dataset for one class: golf club.

Layout follows https://docs.ultralytics.com/datasets/segment

    datasets/golf-club-seg/
      images/train|val/
      labels/train|val/   # <class-index> <x1> <y1> ... normalized polygon
    datasets/golf-club-seg.yaml

Identical copies under images/ are dropped by file hash. Polygons are drafted
with YOLOE-26n prompted for "golf club", because the COCO yolo26n-seg
checkpoint has no golf-club class to copy labels from.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
DATASET_DIR = ROOT / "datasets" / "golf-club-seg"
YAML_PATH = ROOT / "datasets" / "golf-club-seg.yaml"
PREVIEW_DIR = DATASET_DIR / "preview"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAME = "golf club"
CONFIDENCE = 0.15


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


def unique_images(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique = []
    for path in paths:
        digest = file_hash(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
    return unique


def polygon_row(points) -> str | None:
    if len(points) < 3:
        return None
    coords = []
    for x, y in points:
        x = min(1.0, max(0.0, float(x)))
        y = min(1.0, max(0.0, float(y)))
        coords.append(f"{x:.6f} {y:.6f}")
    return "0 " + " ".join(coords)


def reset_dataset() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True)
    PREVIEW_DIR.mkdir(parents=True)


def save_preview(image_path: Path, label_path: Path, destination: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        values = [float(value) for value in parts[1:]]
        polygon = [
            [int(values[i] * width), int(values[i + 1] * height)]
            for i in range(0, len(values), 2)
        ]
        contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [contour], True, (0, 220, 0), 2)
    cv2.imwrite(str(destination), image)


def main() -> None:
    sources = unique_images(collect_images(IMAGE_DIR))
    if not sources:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    model = YOLOE("yoloe-26n-seg.pt")
    model.set_classes([CLASS_NAME])
    results = model.predict(
        source=[str(path) for path in sources],
        conf=CONFIDENCE,
        imgsz=640,
        stream=True,
        verbose=False,
    )

    labeled: list[tuple[Path, list[str]]] = []
    for result in results:
        rows = []
        if result.masks is not None:
            for polygon in result.masks.xyn:
                row = polygon_row(polygon)
                if row is not None:
                    rows.append(row)
        if rows:
            labeled.append((Path(result.path), rows))

    if len(labeled) < 2:
        raise SystemExit(
            f"Only {len(labeled)} image(s) received a golf-club polygon. "
            "A train/val segmentation dataset needs at least two labeled images."
        )

    random.Random(0).shuffle(labeled)
    val_count = max(1, round(len(labeled) * 0.2))
    if val_count >= len(labeled):
        val_count = len(labeled) // 5 or 1
    val_items = labeled[:val_count]
    train_items = labeled[val_count:]

    reset_dataset()
    for split, items in (("train", train_items), ("val", val_items)):
        for source, rows in items:
            image_dest = DATASET_DIR / "images" / split / source.name
            label_dest = DATASET_DIR / "labels" / split / f"{source.stem}.txt"
            shutil.copy2(source, image_dest)
            label_dest.write_text("\n".join(rows) + "\n", encoding="utf-8")

    for index, (source, _) in enumerate(train_items[:4]):
        label_path = DATASET_DIR / "labels" / "train" / f"{source.stem}.txt"
        image_path = DATASET_DIR / "images" / "train" / source.name
        save_preview(image_path, label_path, PREVIEW_DIR / f"train_{index:02d}{source.suffix}")

    YAML_PATH.write_text(
        "\n".join(
            [
                "# Golf-club instance segmentation dataset",
                "# Format: https://docs.ultralytics.com/datasets/segment",
                "# Polygons were drafted by YOLOE-26n with the text prompt 'golf club'.",
                f"path: {DATASET_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: golf club",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Unique source images: {len(sources)}")
    print(f"Labeled images: {len(labeled)} (train {len(train_items)}, val {len(val_items)})")
    print(f"Dataset: {DATASET_DIR}")
    print(f"YAML: {YAML_PATH}")


if __name__ == "__main__":
    main()
