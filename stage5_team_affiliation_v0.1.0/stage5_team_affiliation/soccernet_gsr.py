from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json


def load_gsr_labels(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_track_team_labels(labels: Mapping[str, Any]) -> Dict[str, str]:
    """Extract stable GSR track->team labels from COCO-like annotations.

    SoccerNet GSR team labels are usually strings such as ``left`` and ``right``.
    Referees and annotations without a team are excluded.
    """
    votes: Dict[str, Dict[str, int]] = {}
    for ann in labels.get("annotations", []):
        attrs = ann.get("attributes") or {}
        team = attrs.get("team")
        role = str(attrs.get("role") or "")
        if team is None or role == "referee":
            continue
        tid = str(ann.get("track_id"))
        votes.setdefault(tid, {})[str(team)] = votes.setdefault(tid, {}).get(str(team), 0) + 1
    out: Dict[str, str] = {}
    for tid, counts in votes.items():
        out[tid] = max(counts, key=counts.get)
    return out
