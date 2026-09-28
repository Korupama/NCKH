from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from stage7_game_state.adapters import (
    extract_stage1_view,
    extract_stage5_players,
    extract_stage6_contact,
    load_json,
    normalize_track_id,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _frame_index(payload: Dict[str, Any]) -> Optional[int]:
    value = _first(
        payload.get("frame_index"),
        payload.get("selected_frame_index"),
        _dig(payload, "selected_frame", "frame_index"),
        _dig(payload, "selected_frame_ball", "frame_index"),
        _dig(payload, "metadata", "frame_index"),
        _dig(payload, "frame", "index"),
    )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _camera_status(stage1: Dict[str, Any]) -> Optional[str]:
    value = _first(
        stage1.get("status"),
        stage1.get("camera_status"),
        _dig(stage1, "camera", "status"),
        _dig(stage1, "quality", "status"),
    )
    return str(value).upper() if value is not None else None


def build_preflight(
    stage1_path: str,
    stage5_path: str,
    stage6_path: str,
    *,
    known_referee_tracks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    paths = {"stage1": Path(stage1_path), "stage5": Path(stage5_path), "stage6": Path(stage6_path)}
    sources: Dict[str, Any] = {}
    blockers: List[str] = []
    warnings: List[str] = []

    for name, path in paths.items():
        exists = path.is_file()
        sources[name] = {
            "path": str(path.resolve()),
            "exists": exists,
            "sha256": _sha256(path) if exists else None,
        }
        if not exists:
            blockers.append(f"{name.upper()}_FILE_MISSING")

    if blockers:
        return {
            "status": "BLOCKED",
            "sources": sources,
            "blockers": blockers,
            "warnings": warnings,
            "benchmark_manifest_paths_ready": False,
            "independent_ground_truth_verified": False,
            "production_accuracy_evaluated": False,
        }

    stage1 = load_json(paths["stage1"])
    stage5 = load_json(paths["stage5"])
    stage6 = load_json(paths["stage6"])

    contact = extract_stage6_contact(stage6)
    players = extract_stage5_players(stage5)
    x_view, hit, view_meta = extract_stage1_view(stage1)
    frames = {
        "stage1": _frame_index(stage1),
        "stage5": _frame_index(stage5),
        "stage6": contact.get("frame_index") if contact.get("frame_index") is not None else _frame_index(stage6),
    }
    known_frames = [v for v in frames.values() if v is not None]
    if len(set(known_frames)) > 1:
        blockers.append("FRAME_INDEX_MISMATCH")

    camera_status = _camera_status(stage1)
    if camera_status == "DEGRADED":
        warnings.append("CAMERA_DEGRADED_REVIEW_REQUIRED")
    elif camera_status in {"INVALID", "FAILED", "FAIL"}:
        blockers.append("CAMERA_INVALID")

    if x_view is None:
        if hit is not None and view_meta.get("valid") is False:
            blockers.append("CENTRE_RAY_PITCH_HIT_INVALID")
        else:
            blockers.append("CENTRE_RAY_PITCH_HIT_MISSING")

    by_track = {p["track_id"]: p for p in players}
    toucher_id = contact.get("track_id")
    if toucher_id is None:
        blockers.append("MISSING_CONTACT_TRACK_ID")
        toucher = None
    else:
        toucher = by_track.get(toucher_id)
        if toucher is None:
            blockers.append("TOUCHER_TRACK_NOT_FOUND_IN_STAGE5")
        else:
            if toucher.get("is_referee"):
                blockers.append("TOUCHER_IS_REFEREE")
            if toucher.get("team_key") is None:
                blockers.append("TOUCHER_TEAM_UNRESOLVED")

    known_refs = {normalize_track_id(v) for v in (known_referee_tracks or [])}
    known_refs.discard(None)
    mislabeled_refs = []
    for tid in sorted(known_refs):
        p = by_track.get(tid)
        if p is not None and not p.get("is_referee"):
            mislabeled_refs.append(tid)
    if mislabeled_refs:
        blockers.append("KNOWN_REFEREE_MISLABELLED_UPSTREAM")

    benchmark_paths_ready = all(v["exists"] for v in sources.values()) and not any(
        b.endswith("_FILE_MISSING") for b in blockers
    )

    return {
        "status": "READY" if not blockers else "BLOCKED",
        "sources": sources,
        "frames": frames,
        "camera_status": camera_status,
        "centre_ray_pitch_hit_m": hit,
        "centre_ray": {
            "x_m": x_view,
            "source": view_meta.get("source"),
            "derived": bool(view_meta.get("derived", False)),
            "valid": view_meta.get("valid"),
            "view_pitch_half": view_meta.get("view_pitch_half"),
            "geometry": {k: v for k, v in view_meta.items() if k not in {"source", "derived", "valid", "view_pitch_half"}},
        },
        "contact": contact,
        "num_tracks": len(players),
        "toucher_team_id": toucher.get("team_id") if toucher is not None else None,
        "known_referee_mislabels": mislabeled_refs,
        "blockers": blockers,
        "warnings": warnings,
        "benchmark_manifest_paths_ready": benchmark_paths_ready,
        "independent_ground_truth_verified": False,
        "production_accuracy_evaluated": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 7 integration preflight")
    p.add_argument("--stage1", required=True)
    p.add_argument("--stage5", required=True)
    p.add_argument("--stage6", required=True)
    p.add_argument("--known-referee-track", action="append", default=[])
    p.add_argument("--output")
    args = p.parse_args()

    report = build_preflight(
        args.stage1,
        args.stage5,
        args.stage6,
        known_referee_tracks=args.known_referee_track,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
