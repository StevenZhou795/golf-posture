"""Fine-tune YOLO26n-seg on the golf-club segmentation dataset.

Training call matches https://docs.ultralytics.com/datasets/segment
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "golf-club-seg.yaml"


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(f"Dataset YAML not found: {DATA}. Run prepare_golf_club_seg.py first.")

    model = YOLO("yolo26n-seg.pt")
    model.train(
        data=str(DATA),
        epochs=40,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        cache="ram",
        project=str(ROOT / "runs" / "segment"),
        name="golf_club_train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
