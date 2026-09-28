from __future__ import annotations

"""Build a canonical Stage-6 contact benchmark manifest from official FOOTPASS GT.

FOOTPASS GT identifies an event by (game_key, frame, team, jersey, class).  Stage 6
internally predicts a project track ID, so this builder accepts a small bridge CSV
that points each chosen official event to the corresponding Stage-6 state and,
optionally, to a GT project track ID or track-identity-map JSON.

The builder validates every bridge row against the official FOOTPASS play-by-play
file so a frozen benchmark cannot silently drift away from the source annotations.
"""

from pathlib import Path
from typing import Any
import csv
import json


ACTION_ID_TO_NAME = {
    1: "Drive",
    2: "Pass",
    3: "Cross",
    4: "Shot",
    5: "Header",
    6: "Throw-in",
    7: "Tackle",
    8: "Block",
}
ACTION_NAME_TO_ID = {v.lower(): k for k, v in ACTION_ID_TO_NAME.items()}


def _blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _int_or_none(value: Any) -> int | None:
    value = _blank(value)
    if value is None:
        return None
    return int(float(value))


def _event_index(playbyplay_json: str | Path) -> tuple[Path, dict[tuple[str, int, int, int, int], list[Any]]]:
    path = Path(playbyplay_json).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events") or {}
    out: dict[tuple[str, int, int, int, int], list[Any]] = {}
    for game_key, rows in events.items():
        for event in rows:
            if not isinstance(event, (list, tuple)) or len(event) < 4:
                continue
            frame, team, jersey, action_id = map(int, event[:4])
            out[(str(game_key), frame, team, jersey, action_id)] = list(event)
    return path, out


def _action_id(row: dict[str, str]) -> int | None:
    aid = _int_or_none(row.get("action_id"))
    if aid is not None:
        return aid
    name = _blank(row.get("action_class"))
    if name is None:
        return None
    try:
        return int(float(name))
    except (TypeError, ValueError):
        pass
    return ACTION_NAME_TO_ID.get(str(name).strip().lower())


def _canonical_gt_region(action_id: int, explicit: Any, derive_unambiguous_regions: bool) -> tuple[str | None, str | None]:
    if _blank(explicit) is not None:
        return str(explicit).strip().upper().replace("-", "_").replace(" ", "_"), "BRIDGE_ANNOTATION"
    if not derive_unambiguous_regions:
        return None, None
    # These two classes encode the contacted body family by definition.  We do not
    # infer Pass/Cross/Shot/Drive/Tackle/Block as FOOT because that is not guaranteed.
    if action_id == 5:
        return "HEAD", "FOOTPASS_ACTION_SEMANTICS"
    if action_id == 6:
        return "ARM_HAND", "FOOTPASS_ACTION_SEMANTICS"
    return None, None


def build_footpass_contact_manifest(
    *,
    playbyplay_json: str | Path,
    bridge_csv: str | Path,
    output_json: str | Path,
    frozen: bool = False,
    derive_unambiguous_regions: bool = True,
) -> dict[str, Any]:
    gt_path, gt = _event_index(playbyplay_json)
    bridge_path = Path(bridge_csv).expanduser().resolve()
    with bridge_path.open("r", encoding="utf-8-sig", newline="") as handle:
        bridge = [dict(row) for row in csv.DictReader(handle)]
    if not bridge:
        raise ValueError("Bridge CSV contains no rows")

    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for idx, row in enumerate(bridge):
        game_key = str(_blank(row.get("game_key")) or "")
        frame = _int_or_none(row.get("frame"))
        team = _int_or_none(row.get("team"))
        jersey = _int_or_none(row.get("jersey"))
        action_id = _action_id(row)
        if not game_key or None in (frame, team, jersey, action_id):
            raise ValueError(f"Bridge row {idx + 2}: game_key, frame, team, jersey and action_id/action_class are required")
        key = (game_key, int(frame), int(team), int(jersey), int(action_id))
        if key in seen:
            raise ValueError(f"Duplicate bridge event: {key}")
        seen.add(key)
        if key not in gt:
            raise ValueError(f"Bridge row {idx + 2} does not match official FOOTPASS GT: {key}")

        state_json = _blank(row.get("state_json"))
        if state_json is None:
            raise ValueError(f"Bridge row {idx + 2}: state_json is required")
        normalized_state = str(state_json).strip().replace("\\", "/").lower()
        if frozen and ("relative/path/to/" in normalized_state or "path/to/" in normalized_state or "replace_me" in normalized_state):
            raise ValueError(
                f"Bridge row {idx + 2}: frozen manifest cannot contain placeholder state_json: {state_json}. "
                "Run Stage 6 on the selected FOOTPASS cases and bind the real ball_trajectory_state.json paths first."
            )
        gt_region, region_source = _canonical_gt_region(action_id, row.get("gt_region"), derive_unambiguous_regions)

        case = {
            "case_id": str(_blank(row.get("case_id")) or f"{game_key}_f{frame}_t{team}_j{jersey}_a{action_id}"),
            "source_dataset": "FOOTPASS",
            "split": str(_blank(row.get("split")) or "validation"),
            "game_key": game_key,
            "event_frame": int(frame),
            "gt_team": int(team),
            "gt_jersey": int(jersey),
            "action_id": int(action_id),
            "action_class": ACTION_ID_TO_NAME.get(int(action_id), str(action_id)),
            "state_json": str(state_json),
            "gt_track_id": _blank(row.get("gt_track_id")),
            "track_identity_map_json": _blank(row.get("track_identity_map_json")),
            "gt_region": gt_region,
            "gt_region_source": region_source,
            "gt_contact_binary": _blank(row.get("gt_contact_binary")),
            "gt_x_m": _blank(row.get("gt_x_m")),
            "source_event": gt[key],
        }
        cases.append(case)

    output = Path(output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "name": "stage6-footpass-contact-eval",
            "dataset": "FOOTPASS",
            "split": "validation",
            "frozen": bool(frozen),
            "official_playbyplay_gt": str(gt_path),
            "bridge_csv": str(bridge_path),
            "event_schema": "(frame, team, jersey, class, ...)",
            "note": "FOOTPASS supplies actor/action GT but not metric ball-X. gt_x_m must come from an independent frozen source if production Ball-X is scored.",
        },
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "READY", "output": str(output), "cases": len(cases), "frozen": bool(frozen)}


def write_footpass_bridge_template(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "case_id", "game_key", "frame", "team", "jersey", "action_class",
        "state_json", "gt_track_id", "track_identity_map_json",
        "gt_region", "gt_contact_binary", "gt_x_m", "split",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "case_id": "example_0001",
            "game_key": "game_18_H1",
            "frame": 64,
            "team": 0,
            "jersey": 3,
            "action_class": "Drive",
            "state_json": "relative/path/to/ball_trajectory_state.json",
            "gt_track_id": "",
            "track_identity_map_json": "relative/path/to/track_identity_map.json",
            "gt_region": "",
            "gt_contact_binary": "",
            "gt_x_m": "",
            "split": "validation",
        })
    return target
