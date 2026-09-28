from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import cv2
import numpy as np


_STILL_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_still_image(path: str | Path) -> np.ndarray | None:
    source = Path(path)
    if source.suffix.lower() not in _STILL_IMAGE_SUFFIXES:
        return None
    return cv2.imread(str(source), cv2.IMREAD_COLOR)


def video_metadata(path: str | Path) -> Dict[str, float | int]:
    still = _read_still_image(path)
    if still is not None:
        height, width = still.shape[:2]
        return {"fps": 0.0, "frame_count": 1, "width": int(width), "height": int(height)}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    out = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return out


def read_frames(path: str | Path, frame_indices: Iterable[int]) -> Dict[int, np.ndarray]:
    wanted = sorted(set(int(x) for x in frame_indices))
    if not wanted:
        return {}
    still = _read_still_image(path)
    if still is not None:
        # A live request contains exactly the selected replay frame as an image.
        # Preserve the replay frame index as the dictionary key instead of
        # treating the image as frame zero of a one-frame video.
        return {idx: still.copy() for idx in wanted}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    out: Dict[int, np.ndarray] = {}
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            out[idx] = frame
    cap.release()
    return out
