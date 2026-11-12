"""BlazePose GHUM pose landmarker models (lite, full, heavy)."""
from pathlib import Path
import urllib.request

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{kind}/float16/latest/pose_landmarker_{kind}.task")
# A truncated download is far smaller than any real model (lite is ~5.5 MB).
MIN_MODEL_BYTES = 2_000_000
MODEL_TYPES = ("lite", "full", "heavy")


def model_path(kind: str = "heavy") -> Path:
    """Local path of the `kind` model, downloading it on first use."""
    if kind not in MODEL_TYPES:
        raise ValueError(f"Unknown BlazePose model type {kind!r}; choose one of {', '.join(MODEL_TYPES)}")
    path = MODEL_DIR / f"pose_landmarker_{kind}.task"
    if path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    url = MODEL_URL.format(kind=kind)
    print(f"Downloading BlazePose {kind} model: {url}", flush=True)
    urllib.request.urlretrieve(url, partial)
    if partial.stat().st_size < MIN_MODEL_BYTES:
        size = partial.stat().st_size
        partial.unlink()
        raise RuntimeError(f"Downloaded {kind} model is only {size} bytes; check network access to {url}")
    partial.replace(path)
    return path
