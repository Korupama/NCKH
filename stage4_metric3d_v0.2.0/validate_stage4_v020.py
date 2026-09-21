from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _xyz_map(state: Mapping[str, Any]) -> dict[tuple[str, int, int], object]:
    result = {}
    for track in state.get("tracks") or []:
        track_id = str(track.get("track_id"))
        for observation in track.get("observations") or []:
            frame = int(observation.get("frame_index", -1))
            for joint in observation.get("metric_pose23") or []:
                result[(track_id, frame, int(joint.get("index", -1)))] = joint.get("xyz_world_m")
    return result


def validate_v020(
    *,
    state_path: str | Path,
    base_state_path: str | Path,
    handoff_path: str | Path,
) -> dict[str, Any]:
    state = _load(state_path)
    base = _load(base_state_path)
    handoff = _load(handoff_path)
    current_xyz = _xyz_map(state)
    base_xyz = _xyz_map(base)
    keys = sorted(set(current_xyz).union(base_xyz))
    max_delta = 0.0
    mismatched_xyz_records = 0
    for key in keys:
        before = base_xyz.get(key)
        after = current_xyz.get(key)
        if before is None or after is None:
            if before != after:
                mismatched_xyz_records += 1
            continue
        delta = float(np.max(np.abs(np.asarray(before, dtype=float) - np.asarray(after, dtype=float))))
        max_delta = max(max_delta, delta)
        if delta > 1e-12:
            mismatched_xyz_records += 1

    base_status = {str(track.get("track_id")): str(track.get("selected_frame_pose_status")) for track in base.get("tracks") or []}
    state_status = {str(track.get("track_id")): str(track.get("selected_frame_pose_status")) for track in state.get("tracks") or []}
    pose_status_preserved = base_status == state_status
    eligible = {track_id for track_id, status in base_status.items() if status in {"VALID", "DEGRADED"}}
    uncertainty_by_track = {
        str(track.get("track_id")): str((track.get("selected_frame_uncertainty") or {}).get("status", "NOT_RUN"))
        for track in state.get("tracks") or []
    }
    uncertainty_counts = Counter(uncertainty_by_track.values())
    completed = {track_id for track_id in eligible if uncertainty_by_track.get(track_id) in {"OK", "DEGRADED"}}
    extrema_present = {
        str(track.get("track_id")): bool(
            (((track.get("selected_frame_uncertainty") or {}).get("longitudinal") or {}).get("legal_extrema_x"))
        )
        for track in state.get("tracks") or []
        if str(track.get("track_id")) in eligible
    }
    attacks_applied = [
        track_id for track_id in eligible
        if bool(
            ((next(track for track in state.get("tracks") or [] if str(track.get("track_id")) == track_id).get("selected_frame_uncertainty") or {}).get("attack_direction_applied"))
        )
    ]
    checks = {
        "stage4_version_is_v020": state.get("stage4_version") == "stage4-metric3d-0.2.0",
        "base_xyz_preserved": mismatched_xyz_records == 0 and max_delta <= 1e-12,
        "selected_frame_pose_status_preserved": pose_status_preserved,
        "all_eligible_tracks_have_uncertainty": completed == eligible,
        "all_eligible_tracks_have_legal_extrema_x": all(extrema_present.values()) if extrema_present else False,
        "attack_direction_not_applied": not attacks_applied,
        "uncertainty_not_claimed_calibrated": not bool((state.get("uncertainty_analysis") or {}).get("calibrated", False)),
        "research_accuracy_not_frozen": not bool((state.get("acceptance_gate") or {}).get("research_accuracy_frozen", False)),
        "handoff_schema_1_2": handoff.get("schema_version") == "stage4-to-stage5-handoff-1.2",
    }
    return {
        "schema_version": "stage4-v020-validation-report-1.0",
        "passed": all(checks.values()),
        "checks": checks,
        "base_xyz_comparison": {
            "records_compared": len(keys),
            "mismatched_records": mismatched_xyz_records,
            "max_abs_delta_m": max_delta,
        },
        "pose_status_counts": dict(Counter(state_status.values())),
        "uncertainty_status_counts": dict(uncertainty_counts),
        "eligible_track_ids": sorted(eligible),
        "completed_eligible_track_ids": sorted(completed),
        "incomplete_eligible_track_ids": sorted(eligible - completed),
        "tracks_missing_legal_extrema_x": sorted(track_id for track_id, present in extrema_present.items() if not present),
        "tracks_with_attack_direction_applied": sorted(attacks_applied),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage-4 v0.2 uncertainty without changing its artifacts")
    parser.add_argument("--state", required=True)
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = validate_v020(state_path=args.state, base_state_path=args.base_state, handoff_path=args.handoff)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
