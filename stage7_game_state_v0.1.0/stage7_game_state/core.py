from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .adapters import (
    _dig,
    _first_non_none,
    extract_stage1_view,
    extract_stage5_players,
    extract_stage6_contact,
    load_json,
)
from .models import GameStateContext


def _unique_sorted(values: List[str]) -> List[str]:
    return sorted(set(values), key=lambda x: (len(x), x))


def _evaluate_invariants(ctx: GameStateContext, players: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, bool]]:
    attackers = set(ctx.sets.get("attackers", []))
    opponents = set(ctx.sets.get("opponents", []))
    referees = set(ctx.sets.get("referees_excluded", []))
    eligible_known = {
        p["track_id"] for p in players
        if p["active"] and not p["is_referee"] and p["team_key"] is not None
    }
    toucher_id = ctx.toucher.get("track_id") if isinstance(ctx.toucher, dict) else None
    resolved = ctx.status in {"VALID", "DEGRADED"}
    checks = {
        "toucher_is_attacker": (
            toucher_id in attackers if toucher_id else "ATTACKING_TEAM_INFERRED_FROM_GOALKEEPER" in ctx.reasons
        ) if resolved else True,
        "sets_disjoint": attackers.isdisjoint(opponents),
        "referees_excluded": referees.isdisjoint(attackers | opponents),
        "eligible_partition_complete": (attackers | opponents) == eligible_known if resolved else True,
        "attack_direction_binary": (
            isinstance(ctx.attack_direction, dict) and ctx.attack_direction.get("s") in (-1, 1)
        ) if resolved else True,
    }
    return all(checks.values()), checks


def build_game_state_context(
    stage1_input: Any,
    stage5_input: Any,
    stage6_input: Any,
    *,
    centre_ray_epsilon_m: float = 1e-6,
    allow_goalkeeper_fallback: bool = False,
) -> GameStateContext:
    stage1 = load_json(stage1_input)
    stage5 = load_json(stage5_input)
    stage6 = load_json(stage6_input)

    ctx = GameStateContext()
    contact = extract_stage6_contact(stage6)
    players = extract_stage5_players(stage5)
    x_view, hit, view_meta = extract_stage1_view(stage1)
    ctx.frame_index = contact.get("frame_index")

    diagnostics: Dict[str, Any] = {
        "num_stage5_players": len(players),
        "contact_status": contact.get("status"),
        "contact_region": contact.get("region"),
        "contact_confidence": contact.get("confidence"),
        "contact_track_id_confirmed": contact.get("track_id"),
        "contact_nearest_track_id": contact.get("nearest_track_id"),
        "contact_spatial_distance_px": contact.get("image_distance_px"),
        "contact_spatial_threshold_px": contact.get("threshold_px"),
        "centre_ray_pitch_hit_m": hit,
        "centre_ray_source": view_meta.get("source"),
        "centre_ray_derived": bool(view_meta.get("derived", False)),
        "centre_ray_valid": view_meta.get("valid"),
        "view_pitch_half": view_meta.get("view_pitch_half"),
        "centre_ray_geometry": {k: v for k, v in view_meta.items() if k not in {"source", "derived", "valid", "view_pitch_half"}},
    }
    reasons: List[str] = []
    # Direction uses the sign of the plane intersection, even beyond the pitch.
    # Retain Stage 1's geometry validity; do not promote the hit to valid geometry.
    outside_pitch_direction = bool(
        hit is not None
        and view_meta.get("geometry_reason") == "CENTRE_RAY_PITCH_HIT_OUT_OF_BOUNDS"
    )
    if outside_pitch_direction:
        x_view = hit[0]
        reasons.append("ATTACK_DIRECTION_FROM_OUT_OF_BOUNDS_CENTRE_RAY")
    diagnostics["direction_policy"] = "CENTRE_RAY_X_SIGN_ALLOW_OUT_OF_BOUNDS"

    confirmed_toucher_id = contact.get("track_id")
    tentative_contact = bool(
        not confirmed_toucher_id
        and contact.get("status") == "INSUFFICIENT_TEMPORAL_SUPPORT"
        and contact.get("nearest_track_id")
    )
    diagnostics["contact_tentative_spatial_only"] = tentative_contact
    toucher_id = contact.get("nearest_track_id") if tentative_contact else confirmed_toucher_id
    if not toucher_id:
        reasons.append("MISSING_CONTACT_TRACK_ID")
    elif tentative_contact:
        reasons.append("CONTACT_TENTATIVE_SPATIAL_ONLY")

    by_track = {p["track_id"]: p for p in players}
    toucher = by_track.get(toucher_id) if toucher_id else None
    if toucher_id and toucher is None:
        reasons.append("TOUCHER_TRACK_NOT_FOUND_IN_STAGE5")
    if toucher is not None and toucher["is_referee"]:
        reasons.append("TOUCHER_IS_REFEREE")
    if toucher is not None and not toucher["active"]:
        reasons.append("TOUCHER_NOT_ACTIVE_AT_T0")
    if toucher is not None and toucher["team_key"] is None:
        reasons.append("TOUCHER_TEAM_UNRESOLVED")

    # Multi-tier attack direction resolution:
    # Tier 1: Primary - Stage-1 camera optical centre-ray pitch hit X coordinate
    # Tier 2: Secondary - Explicit attack direction from Stage 1 or Stage 6 metadata
    # Tier 3: Tertiary - View pitch half label ("LEFT" or "RIGHT")
    resolved_s = None
    direction_source = None
    if x_view is not None and abs(x_view) > centre_ray_epsilon_m:
        resolved_s = 1 if x_view > 0 else -1
        direction_source = view_meta.get("source") or "stage1.centre_ray_pitch_hit"
    else:
        # Check explicit upstream attack direction
        explicit_s = _first_non_none(
            stage1.get("attack_direction_s"),
            _dig(stage1, ["attack_direction", "s"]),
            _dig(stage6, ["stage7", "attack_direction_s"]),
            stage6.get("attack_direction_s"),
            _dig(stage5, ["attack_direction_s"]),
        )
        if explicit_s in (-1, 1, "-1", "1"):
            resolved_s = int(explicit_s)
            direction_source = "upstream_explicit_attack_direction"

        # Check view pitch half from calibrated camera
        if resolved_s is None:
            v_half = str(view_meta.get("view_pitch_half") or _dig(stage1, ["view", "view_pitch_half"]) or "").upper()
            if v_half == "RIGHT":
                resolved_s = 1
                direction_source = "stage1.view_pitch_half_label"
            elif v_half == "LEFT":
                resolved_s = -1
                direction_source = "stage1.view_pitch_half_label"

    if resolved_s is None:
        if x_view is None:
            if hit is not None and view_meta.get("valid") is False:
                reasons.append("CENTRE_RAY_PITCH_HIT_INVALID")
            else:
                reasons.append("CENTRE_RAY_PITCH_HIT_MISSING")
        elif abs(x_view) <= centre_ray_epsilon_m:
            reasons.append("CENTRE_RAY_X_AMBIGUOUS")
    elif x_view is None or abs(x_view) <= centre_ray_epsilon_m:
        reasons.append("ATTACK_DIRECTION_RESOLVED_VIA_FALLBACK")

    attacking_team_key = toucher["team_key"] if toucher is not None else None
    attacking_team_raw = toucher["team_id"] if toucher is not None else None
    goalkeeper_fallback = False
    # A single visible, affiliated keeper may identify the defending team in
    # the viewed attacking half. This is an explicit demo assumption, not contact evidence.
    if allow_goalkeeper_fallback and not toucher_id and resolved_s in (-1, 1):
        eligible = [p for p in players if p["active"] and not p["is_referee"] and p["team_key"] is not None]
        visible_keepers = [p for p in players if p["active"] and not p["is_referee"]
                           and str(p["role"]).lower() in {"goalkeeper", "keeper", "gk"}]
        keepers = [p for p in visible_keepers if p["team_key"] is not None
                   and p["raw"].get("team_status", "VALID") == "VALID"
                   and p["raw"].get("assignment_method") != "OUTSIDE_PITCH_EXCLUDED"]
        teams = {p["team_key"] for p in eligible}
        if len(visible_keepers) == 1 and len(keepers) == 1 and len(teams) == 2:
            keeper = keepers[0]
            attacking_team_key = next(key for key in teams if key != keeper["team_key"])
            attacking_team_raw = next(p["team_id"] for p in eligible if p["team_key"] == attacking_team_key)
            goalkeeper_fallback = True
            reasons.append("ATTACKING_TEAM_INFERRED_FROM_GOALKEEPER")
            diagnostics["team_resolution"] = {
                "source": "VISIBLE_GOALKEEPER_DEFENDING_TEAM_ASSUMPTION",
                "goalkeeper_track_id": keeper["track_id"],
                "defending_team_id": keeper["team_id"],
                "reference_only": True,
            }

    attackers: List[str] = []
    opponents: List[str] = []
    referees: List[str] = []
    unknown_team: List[str] = []
    inactive: List[str] = []

    if attacking_team_key is not None:
        for p in players:
            if not p["active"]:
                inactive.append(p["track_id"])
                continue
            if p["is_referee"]:
                referees.append(p["track_id"])
                continue
            if p["team_key"] is None:
                unknown_team.append(p["track_id"])
                continue
            if p["team_key"] == attacking_team_key:
                attackers.append(p["track_id"])
            else:
                opponents.append(p["track_id"])

    if attacking_team_key is not None and not opponents:
        reasons.append("NO_OPPONENTS_RESOLVED")
    if attacking_team_key is not None and not attackers:
        reasons.append("NO_ATTACKERS_RESOLVED")

    ctx.toucher = None if toucher is None else {
        "track_id": toucher["track_id"],
        "team_id": toucher["team_id"],
        "role": toucher["role"],
        "source": contact["source"],
        "evidence_status": contact.get("status"),
        "evidence_level": "TENTATIVE_SPATIAL_ONLY" if tentative_contact else "CONFIRMED",
        "image_distance_px": contact.get("image_distance_px"),
        "threshold_px": contact.get("threshold_px"),
    }
    ctx.attacking_team_id = attacking_team_raw

    if resolved_s is not None:
        ctx.attack_direction = {
            "s": resolved_s,
            "label": "LEFT_TO_RIGHT" if resolved_s == 1 else "RIGHT_TO_LEFT",
            "centre_ray_x_m": float(x_view) if x_view is not None else None,
            "view_pitch_half": ("RIGHT" if resolved_s == 1 else "LEFT") if outside_pitch_direction else view_meta.get("view_pitch_half"),
            "source": direction_source,
        }

    ctx.sets = {
        "attackers": _unique_sorted(attackers),
        "opponents": _unique_sorted(opponents),
        "referees_excluded": _unique_sorted(referees),
        "unknown_team_excluded": _unique_sorted(unknown_team),
        "inactive_excluded": _unique_sorted(inactive),
    }

    hard_reasons = {
        "MISSING_CONTACT_TRACK_ID",
        "TOUCHER_TRACK_NOT_FOUND_IN_STAGE5",
        "TOUCHER_IS_REFEREE",
        "TOUCHER_NOT_ACTIVE_AT_T0",
        "TOUCHER_TEAM_UNRESOLVED",
        "CENTRE_RAY_PITCH_HIT_MISSING",
        "CENTRE_RAY_PITCH_HIT_INVALID",
        "CENTRE_RAY_X_AMBIGUOUS",
        "NO_OPPONENTS_RESOLVED",
        "NO_ATTACKERS_RESOLVED",
    }
    if goalkeeper_fallback:
        hard_reasons.discard("MISSING_CONTACT_TRACK_ID")
    if any(r in hard_reasons for r in reasons):
        ctx.status = "UNRESOLVED"
    elif tentative_contact or outside_pitch_direction or goalkeeper_fallback:
        ctx.status = "DEGRADED"
    else:
        ctx.status = "VALID"
    ctx.reasons = reasons

    inv_ok, inv = _evaluate_invariants(ctx, players)
    diagnostics["invariants"] = inv
    diagnostics["invariant_pass"] = inv_ok
    diagnostics["num_attackers"] = len(ctx.sets["attackers"])
    diagnostics["num_opponents"] = len(ctx.sets["opponents"])
    diagnostics["num_referees_excluded"] = len(ctx.sets["referees_excluded"])
    diagnostics["num_unknown_team_excluded"] = len(ctx.sets["unknown_team_excluded"])
    diagnostics["num_inactive_excluded"] = len(ctx.sets["inactive_excluded"])
    ctx.diagnostics = diagnostics

    if ctx.status in {"VALID", "DEGRADED"} and not inv_ok:
        ctx.status = "UNRESOLVED"
        ctx.reasons.append("INVARIANT_FAILURE")
    return ctx
