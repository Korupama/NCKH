from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import json
import numpy as np


@dataclass
class InitializerObservation:
    track_id: str
    frame_index: int
    backend: str
    raw_keypoints_133: np.ndarray
    keypoint_scores_133: np.ndarray
    relative_depth_133: np.ndarray
    raw: Mapping[str, Any]

    @property
    def relative_depth_23(self) -> np.ndarray:
        return self.relative_depth_133[:23].astype(np.float64, copy=True)

    @property
    def scores_23(self) -> np.ndarray:
        return self.keypoint_scores_133[:23].astype(np.float64, copy=True)


class InitializerCache:
    def __init__(self, records: Mapping[Tuple[str, int], InitializerObservation], path: str | Path):
        self.records = dict(records)
        self.path = str(Path(path).resolve())

    def get(self, track_id: str, frame_index: int) -> Optional[InitializerObservation]:
        return self.records.get((str(track_id), int(frame_index)))

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "InitializerCache":
        path = Path(path).expanduser().resolve()
        records: Dict[Tuple[str, int], InitializerObservation] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "manifest":
                    continue
                tid = str(data["track_id"])
                frame = int(data["frame_index"])
                raw = np.asarray(data["raw_keypoints_133"], dtype=np.float64)
                if raw.shape != (133, 3):
                    raise ValueError(f"Initializer record line {line_no}: expected (133,3), got {raw.shape}")
                scores = np.asarray(data.get("keypoint_scores_133", np.ones(133)), dtype=np.float64).reshape(133)
                rel = data.get("relative_depth_133")
                if rel is None:
                    root = float(np.nanmean(raw[[11, 12], 2]))
                    rel_arr = raw[:, 2] - root
                else:
                    rel_arr = np.asarray(rel, dtype=np.float64).reshape(133)
                key = (tid, frame)
                if key in records:
                    raise ValueError(f"Duplicate initializer record {key}")
                records[key] = InitializerObservation(
                    track_id=tid,
                    frame_index=frame,
                    backend=str(data.get("backend", "unknown")),
                    raw_keypoints_133=raw,
                    keypoint_scores_133=scores,
                    relative_depth_133=rel_arr,
                    raw=data,
                )
        return cls(records, path)


def write_jsonl(path: str | Path, manifest: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "manifest", **dict(manifest)}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return path
