from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import json

RAW_PIXEL_SPACE = "RAW_DISTORTED_PIXEL"


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage3_state(path: str | Path) -> Dict[str, Any]:
    state = load_json(path)
    if state.get("schema_version") != "tracked-pose-2d-state-1.0":
        raise ValueError(f"Unsupported Stage3 schema: {state.get('schema_version')!r}")
    if state.get("coordinate_space") != RAW_PIXEL_SPACE:
        raise ValueError("Stage5 requires RAW_DISTORTED_PIXEL coordinates from Stage3")
    names = ((state.get("keypoint_schema") or {}).get("names") or [])
    if len(names) != 133:
        raise ValueError(f"Expected COCO WholeBody133 keypoint schema; got {len(names)} points")
    return state


def load_stage2_state(path: str | Path | None) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    state = load_json(path)
    if state.get("schema_version") != "entity-track-state-1.0":
        raise ValueError(f"Unsupported Stage2 schema: {state.get('schema_version')!r}")
    rc = state.get("replay_context") or {}
    if rc.get("coordinate_space") != RAW_PIXEL_SPACE:
        raise ValueError("Stage5 requires RAW_DISTORTED_PIXEL Stage2 coordinates")
    return state


def stage3_track_index(state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(t["track_id"]): dict(t) for t in state.get("tracks", [])}


def stage2_track_index(state: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not state:
        return {}
    return {str(t["track_id"]): dict(t) for t in state.get("tracks", [])}


def observation_by_frame(track: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(o["frame_index"]): dict(o) for o in track.get("observations", [])}


def validate_stage2_stage3_alignment(stage2: Optional[Mapping[str, Any]], stage3: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "stage2_provided": stage2 is not None,
        "selected_frame_match": True,
        "image_size_match": True,
        "missing_stage2_track_ids": [],
        "role_mismatches": [],
    }
    if stage2 is None:
        return diagnostics
    r2 = stage2.get("replay_context") or {}
    r3 = stage3.get("replay_context") or {}
    diagnostics["selected_frame_match"] = int(r2.get("selected_frame", -1)) == int(r3.get("selected_frame", -2))
    diagnostics["image_size_match"] = (
        int(r2.get("image_width", -1)), int(r2.get("image_height", -1))
    ) == (
        int(r3.get("image_width", -2)), int(r3.get("image_height", -2))
    )
    idx2 = stage2_track_index(stage2)
    for t3 in stage3.get("tracks", []):
        tid = str(t3["track_id"])
        if tid not in idx2:
            diagnostics["missing_stage2_track_ids"].append(tid)
            continue
        r_up = str(t3.get("upstream_role") or "")
        r2v = str(idx2[tid].get("role") or "")
        if r_up and r2v and r_up != r2v:
            diagnostics["role_mismatches"].append({"track_id": tid, "stage3": r_up, "stage2": r2v})
    diagnostics["ready"] = bool(
        diagnostics["selected_frame_match"]
        and diagnostics["image_size_match"]
        and not diagnostics["role_mismatches"]
    )
    return diagnostics
