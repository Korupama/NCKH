from __future__ import annotations

"""Safety-first helpers for the Stage 5 v0.2.3 residual-role policy.

This module deliberately contains no dataset or GT dependencies.  It operates on
the diagnostics already produced by the role-blind pipeline, so it can be reused
by both the runtime and the TRAIN-only calibrator.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    context = record.get("goal_context")
    if isinstance(context, Mapping) and key in context:
        return context[key]
    return record.get(key, default)


@dataclass(frozen=True)
class GoalkeeperRolePolicy:
    max_goal_distance_m: float = 18.0
    min_top2_rate: float = 0.60
    min_observations: int = 5
    min_rank_frames: int = 5
    ambiguity_margin_m: float = 0.50

    def validate(self) -> None:
        if self.max_goal_distance_m <= 0.0:
            raise ValueError("max_goal_distance_m must be positive")
        if not 0.0 <= self.min_top2_rate <= 1.0:
            raise ValueError("min_top2_rate must be in [0, 1]")
        if self.min_observations < 1 or self.min_rank_frames < 1:
            raise ValueError("observation thresholds must be positive")
        if self.ambiguity_margin_m < 0.0:
            raise ValueError("ambiguity_margin_m must be non-negative")


@dataclass(frozen=True)
class ResidualRolePolicy:
    player_min_margin: float = 0.15
    player_max_distance: float = 0.50
    referee_max_margin: float = 0.10
    referee_min_distance: float = 0.40
    referee_min_goal_distance_m: float = 22.0
    referee_max_top2_rate: float = 0.20

    def validate(self) -> None:
        if self.player_min_margin < 0.0 or self.referee_max_margin < 0.0:
            raise ValueError("appearance margins must be non-negative")
        if self.player_max_distance < 0.0 or self.referee_min_distance < 0.0:
            raise ValueError("appearance distances must be non-negative")
        if self.referee_min_goal_distance_m < 0.0:
            raise ValueError("referee_min_goal_distance_m must be non-negative")
        if not 0.0 <= self.referee_max_top2_rate <= 1.0:
            raise ValueError("referee_max_top2_rate must be in [0, 1]")


def _goal_side(record: Mapping[str, Any]) -> str:
    value = _value(record, "goal_side")
    if value is None:
        value = _value(record, "defended_goal_side")
    if value is None:
        sign = _finite_float(_value(record, "goal_sign"))
        if sign is not None:
            return "RIGHT" if sign > 0 else "LEFT"
    return str(value or "UNKNOWN").upper()


def _track_key(record: Mapping[str, Any]) -> str:
    return str(record.get("track_id", record.get("id", "")))


def goalkeeper_candidates(
    records: Sequence[Mapping[str, Any]],
    policy: GoalkeeperRolePolicy,
) -> list[Mapping[str, Any]]:
    """Return candidates that pass all independent goalkeeper geometry gates.

    The deepest-track selection is intentionally performed *after* these gates.
    This prevents a sparsely observed residual from suppressing a well-supported
    goalkeeper candidate, which was the semantic bug exposed by the TRAIN run.
    """

    policy.validate()
    result: list[Mapping[str, Any]] = []
    for record in records:
        observations = int(_finite_float(_value(record, "observations")) or 0)
        rank_frames = int(_finite_float(_value(record, "rank_frames")) or 0)
        top2_rate = _finite_float(_value(record, "top2_rate"))
        goal_distance = _finite_float(_value(record, "median_goal_distance_m"))
        if observations < policy.min_observations:
            continue
        if rank_frames < policy.min_rank_frames:
            continue
        if top2_rate is None or top2_rate < policy.min_top2_rate:
            continue
        if goal_distance is None or goal_distance > policy.max_goal_distance_m:
            continue
        result.append(record)
    return result


def select_goalkeepers_by_geometry(
    records: Sequence[Mapping[str, Any]],
    policy: GoalkeeperRolePolicy,
) -> dict[str, dict[str, Any]]:
    """Select at most one goalkeeper per sequence and defended goal side.

    Ambiguous depth ordering is an abstention, not a forced assignment.
    Returned decisions are keyed by track id and contain only prediction-time
    diagnostics; no GT role or team field is consulted.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in goalkeeper_candidates(records, policy):
        sequence_id = str(record.get("sequence_id", record.get("sequence", "")))
        groups.setdefault((sequence_id, _goal_side(record)), []).append(record)

    decisions: dict[str, dict[str, Any]] = {}
    for (sequence_id, goal_side), candidates in groups.items():
        ranked = sorted(
            candidates,
            key=lambda row: (
                _finite_float(_value(row, "median_goal_distance_m"))
                if _finite_float(_value(row, "median_goal_distance_m")) is not None
                else float("inf"),
                -(_finite_float(_value(row, "top2_rate")) or 0.0),
                _track_key(row),
            ),
        )
        winner = ranked[0]
        winner_distance = _finite_float(_value(winner, "median_goal_distance_m"))
        runner_distance = (
            _finite_float(_value(ranked[1], "median_goal_distance_m"))
            if len(ranked) > 1
            else None
        )
        if (
            winner_distance is not None
            and runner_distance is not None
            and runner_distance - winner_distance < policy.ambiguity_margin_m
        ):
            reason = "AMBIGUOUS_GOAL_DEPTH_ORDER"
            for row in ranked[:2]:
                decisions[_track_key(row)] = {
                    "role": "UNKNOWN",
                    "reason": reason,
                    "sequence_id": sequence_id,
                    "goal_side": goal_side,
                }
            continue
        decisions[_track_key(winner)] = {
            "role": "GOALKEEPER",
            "reason": "GOALKEEPER_GATES_PASSED",
            "sequence_id": sequence_id,
            "goal_side": goal_side,
            "median_goal_distance_m": winner_distance,
        }
    return decisions


def classify_residual_role(
    record: Mapping[str, Any],
    policy: ResidualRolePolicy,
) -> dict[str, Any]:
    """Classify a non-goalkeeper residual using appearance *and* geometry.

    Appearance alone may recover a confident outfield player.  Referee recovery
    additionally requires the track to stay away from the defended goal and to
    have a low deepest/top-2 rate.  Otherwise the function abstains.
    """

    policy.validate()
    margin = _finite_float(record.get("team_margin", record.get("appearance_margin")))
    distance = _finite_float(
        record.get("nearest_team_distance", record.get("appearance_distance"))
    )
    goal_distance = _finite_float(_value(record, "median_goal_distance_m"))
    top2_rate = _finite_float(_value(record, "top2_rate"))

    if margin is not None and distance is not None:
        if margin >= policy.player_min_margin and distance <= policy.player_max_distance:
            return {"role": "PLAYER", "reason": "APPEARANCE_TEAM_CORE_RECOVERY"}
        referee_geometry = (
            goal_distance is not None
            and goal_distance >= policy.referee_min_goal_distance_m
            and top2_rate is not None
            and top2_rate <= policy.referee_max_top2_rate
        )
        if (
            margin <= policy.referee_max_margin
            and distance >= policy.referee_min_distance
            and referee_geometry
        ):
            return {"role": "REFEREE", "reason": "APPEARANCE_GEOMETRY_REFEREE_RECOVERY"}
    return {"role": "UNKNOWN", "reason": "RESIDUAL_POLICY_ABSTAIN"}


def research_freeze_gate(
    *,
    complete_train_split: bool,
    goalkeeper_role_feasible: bool,
    residual_role_feasible: bool,
    goalkeeper_team_feasible: bool,
) -> dict[str, Any]:
    """Single source of truth for whether a calibrated config may be frozen."""

    checks = {
        "complete_train_split": bool(complete_train_split),
        "goalkeeper_role_feasible": bool(goalkeeper_role_feasible),
        "residual_role_feasible": bool(residual_role_feasible),
        "goalkeeper_team_feasible": bool(goalkeeper_team_feasible),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def retain_minimum_core(
    distances: Iterable[float],
    candidate_indices: Sequence[int],
    min_core_tracks: int,
) -> tuple[list[int], dict[str, Any]]:
    """Deterministic last-resort core retention for robust K=2 fitting.

    The caller supplies each candidate's distance to its assigned centroid.  If
    residual trimming would leave too few samples, the nearest candidates are
    retained instead of marking the whole sequence unavailable.  This helper
    never invents labels and should still be followed by centroid-separation and
    two-cluster occupancy checks in the caller.
    """

    indexed = [
        (float(distance), int(index))
        for distance, index in zip(distances, candidate_indices)
        if isfinite(float(distance))
    ]
    indexed.sort(key=lambda item: (item[0], item[1]))
    keep_count = min(max(int(min_core_tracks), 0), len(indexed))
    kept = sorted(index for _, index in indexed[:keep_count])
    return kept, {
        "used": len(candidate_indices) < min_core_tracks and bool(kept),
        "reason": "MINIMUM_CORE_NEAREST_CENTROID",
        "requested": int(min_core_tracks),
        "available": len(indexed),
        "retained": len(kept),
    }

