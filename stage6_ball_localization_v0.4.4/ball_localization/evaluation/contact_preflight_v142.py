from __future__ import annotations

"""Preflight validation for Stage-6 contact/production benchmark manifests.

Protocol v1.4.2.  This module checks that the frozen benchmark manifest is bound to
real Stage-6 prediction artifacts before contact-production is executed.  It never
creates predictions and never turns missing GT into a PASS.
"""

from pathlib import Path
from typing import Any
import json


_PLACEHOLDER_TOKENS = (
    "relative/path/to/",
    "path/to/",
    "replace_me",
    "todo",
    "example_0001",
)


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _is_placeholder(value: Any) -> bool:
    if _blank(value):
        return False
    text = str(value).strip().replace("\\", "/").lower()
    return any(token in text for token in _PLACEHOLDER_TOKENS)


def _resolve(base: Path, value: Any) -> Path | None:
    if _blank(value):
        return None
    p = Path(str(value)).expanduser()
    if not p.is_absolute():
        p = base.parent / p
    return p.resolve()


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    manifest = Path(path).expanduser().resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return manifest, {}, [dict(x) for x in payload]
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Preflight expects JSON {metadata,cases} or a JSON list")
    return manifest, dict(payload.get("metadata") or {}), [dict(x) for x in payload["cases"]]


def preflight_contact_manifest(manifest: str | Path) -> dict[str, Any]:
    manifest_path, metadata, cases = _load_manifest(manifest)

    issues: list[dict[str, Any]] = []
    state_ok = 0
    track_map_ok = 0
    contact_gt_count = 0
    region_gt_count = 0
    ball_x_gt_count = 0

    for idx, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{idx:05d}")
        state_value = case.get("state_json")
        state_path = _resolve(manifest_path, state_value)

        if _blank(state_value):
            issues.append({"case_id": case_id, "kind": "MISSING_STATE_PATH", "value": None})
        elif _is_placeholder(state_value):
            issues.append({"case_id": case_id, "kind": "PLACEHOLDER_STATE_PATH", "value": str(state_value)})
        elif state_path is None or not state_path.is_file():
            issues.append({"case_id": case_id, "kind": "STATE_FILE_NOT_FOUND", "value": None if state_path is None else str(state_path)})
        else:
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                selected = payload.get("selected_frame_ball") if isinstance(payload, dict) else None
                if not isinstance(selected, dict):
                    issues.append({"case_id": case_id, "kind": "STATE_SCHEMA_MISSING_SELECTED_FRAME_BALL", "value": str(state_path)})
                else:
                    state_ok += 1
            except Exception as exc:  # preflight should report malformed artifacts, not crash at first one
                issues.append({"case_id": case_id, "kind": "STATE_JSON_INVALID", "value": str(state_path), "detail": str(exc)})

        gt_track = case.get("gt_track_id")
        gt_team = case.get("gt_team")
        gt_jersey = case.get("gt_jersey")
        track_map_value = case.get("track_identity_map_json")
        track_map_path = _resolve(manifest_path, track_map_value)

        has_direct_track_gt = not _blank(gt_track)
        has_actor_gt = not _blank(gt_team) and not _blank(gt_jersey)
        has_identity_bridge = has_direct_track_gt

        if has_direct_track_gt:
            contact_gt_count += 1
        elif has_actor_gt:
            if _blank(track_map_value):
                issues.append({"case_id": case_id, "kind": "MISSING_TRACK_IDENTITY_MAP", "value": None})
            elif _is_placeholder(track_map_value):
                issues.append({"case_id": case_id, "kind": "PLACEHOLDER_TRACK_IDENTITY_MAP", "value": str(track_map_value)})
            elif track_map_path is None or not track_map_path.is_file():
                issues.append({"case_id": case_id, "kind": "TRACK_IDENTITY_MAP_NOT_FOUND", "value": None if track_map_path is None else str(track_map_path)})
            else:
                try:
                    json.loads(track_map_path.read_text(encoding="utf-8"))
                    track_map_ok += 1
                    has_identity_bridge = True
                except Exception as exc:
                    issues.append({"case_id": case_id, "kind": "TRACK_IDENTITY_MAP_INVALID", "value": str(track_map_path), "detail": str(exc)})
            if has_identity_bridge:
                contact_gt_count += 1
        else:
            issues.append({"case_id": case_id, "kind": "MISSING_CONTACT_IDENTITY_GT", "value": None})

        gt_region = case.get("gt_contact_binary") or case.get("gt_region")
        if not _blank(gt_region):
            region_gt_count += 1

        gt_x = case.get("gt_x_m")
        gt_xyz = case.get("gt_xyz_m")
        if not _blank(gt_x) or (isinstance(gt_xyz, (list, tuple)) and len(gt_xyz) > 0):
            ball_x_gt_count += 1

    blocking_state_issues = {
        "MISSING_STATE_PATH", "PLACEHOLDER_STATE_PATH", "STATE_FILE_NOT_FOUND",
        "STATE_SCHEMA_MISSING_SELECTED_FRAME_BALL", "STATE_JSON_INVALID",
    }
    blocking_identity_issues = {
        "MISSING_TRACK_IDENTITY_MAP", "PLACEHOLDER_TRACK_IDENTITY_MAP",
        "TRACK_IDENTITY_MAP_NOT_FOUND", "TRACK_IDENTITY_MAP_INVALID",
        "MISSING_CONTACT_IDENTITY_GT",
    }

    state_issue_count = sum(i["kind"] in blocking_state_issues for i in issues)
    identity_issue_count = sum(i["kind"] in blocking_identity_issues for i in issues)

    return {
        "schema_version": "stage6-contact-preflight-1.0",
        "status": "READY" if state_issue_count == 0 else "NOT_READY",
        "manifest": str(manifest_path),
        "frozen": bool(metadata.get("frozen", False)),
        "cases": len(cases),
        "prediction_artifacts": {
            "valid_state_files": state_ok,
            "state_issue_count": state_issue_count,
            "ready_for_contact_production_run": bool(cases and state_issue_count == 0),
        },
        "metric_readiness": {
            "contact_identity": {
                "gt_bound_cases": contact_gt_count,
                "identity_issue_count": identity_issue_count,
                "ready": bool(cases and state_issue_count == 0 and contact_gt_count > 0 and identity_issue_count == 0),
            },
            "foot_vs_nonfoot": {
                "gt_labeled_cases": region_gt_count,
                "ready": bool(cases and state_issue_count == 0 and region_gt_count > 0),
            },
            "production_ball_x": {
                "gt_labeled_cases": ball_x_gt_count,
                "ready": bool(cases and state_issue_count == 0 and ball_x_gt_count > 0),
                "note": "FOOTPASS does not provide independent metric ball-X; gt_x_m/gt_xyz_m must come from a separate frozen GT source.",
            },
        },
        "track_identity_maps_valid": track_map_ok,
        "issues": issues,
    }
