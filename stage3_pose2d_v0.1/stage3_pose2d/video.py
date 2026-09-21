from __future__ import annotations
from pathlib import Path
import cv2


def read_frame(video_path: str | Path, frame_index: int):
    path = Path(video_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
    return frame
