from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .soccernet_gsr import SoccerNetSequence, CANDIDATE_ROLES


@dataclass(frozen=True)
class BenchmarkProtocol:
    name: str
    target_fractions: tuple[float, ...]
    max_sequences: int | None
    half_window_seconds: float = 1.0

    @classmethod
    def named(cls, name: str, *, half_window_seconds: float = 1.0) -> "BenchmarkProtocol":
        n = name.lower()
        if n == "quick":
            return cls("quick", (0.25, 0.50, 0.75), 10, half_window_seconds)
        if n == "full":
            return cls("full", (1/6, 1/3, 1/2, 2/3, 5/6), None, half_window_seconds)
        raise ValueError("protocol must be 'quick' or 'full'")


@dataclass(frozen=True)
class WindowSpec:
    sequence_id: str
    target_frame: int
    window_start: int
    window_end: int
    fps: float

    @property
    def frame_indices(self) -> List[int]:
        return list(range(self.window_start, self.window_end + 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "target_frame": self.target_frame,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "fps": self.fps,
        }


def _nearest_eligible(seq: SoccerNetSequence, desired: int, margin: int) -> int:
    lo = margin
    hi = seq.num_frames - 1 - margin
    eligible = [
        fi for fi in seq.labelled_frames
        if lo <= fi <= hi and any(x.role in CANDIDATE_ROLES for x in seq.gt(fi))
    ]
    if not eligible:
        eligible = [fi for fi in seq.labelled_frames if any(x.role in CANDIDATE_ROLES for x in seq.gt(fi))]
    if not eligible:
        raise ValueError(f"{seq.sequence_id}: no labelled frame with Player/GK GT")
    return min(eligible, key=lambda x: (abs(x - desired), x))


def build_protocol_windows(seq: SoccerNetSequence, protocol: BenchmarkProtocol) -> List[WindowSpec]:
    margin = max(1, int(round(protocol.half_window_seconds * seq.fps)))
    windows: List[WindowSpec] = []
    used = set()
    for frac in protocol.target_fractions:
        desired = int(round(frac * max(0, seq.num_frames - 1)))
        t0 = _nearest_eligible(seq, desired, margin)
        if t0 in used:
            continue
        used.add(t0)
        start = max(0, t0 - margin)
        end = min(seq.num_frames - 1, t0 + margin)
        windows.append(WindowSpec(seq.sequence_id, t0, start, end, seq.fps))
    return windows
