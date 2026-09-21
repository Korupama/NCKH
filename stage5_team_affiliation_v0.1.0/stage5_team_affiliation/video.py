from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import cv2
import numpy as np


def video_metadata(path: str | Path) -> Dict[str, float | int]:
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
