from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_TIE_EPSILON_M = 1e-9


def rank_opponents(extents: Iterable[Dict[str, Any]], *, tie_epsilon_m: float = DEFAULT_TIE_EPSILON_M) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in extents if row.get("goalward_q_m") is not None]
    rows.sort(key=lambda row: (-float(row["goalward_q_m"]), str(row.get("track_id"))))
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        row["rank"] = i
        row["rank_tie_with_previous"] = bool(
            i > 1 and abs(float(row["goalward_q_m"]) - float(rows[i - 2]["goalward_q_m"])) <= tie_epsilon_m
        )
        out.append(row)
    return out


def select_second_last(ranking: List[Dict[str, Any]], *, tie_epsilon_m: float = DEFAULT_TIE_EPSILON_M) -> Dict[str, Any] | None:
    if len(ranking) < 2:
        return None
    q_second = float(ranking[1]["goalward_q_m"])
    candidates = [
        str(row.get("track_id"))
        for row in ranking
        if abs(float(row["goalward_q_m"]) - q_second) <= tie_epsilon_m
    ]
    candidates = sorted(dict.fromkeys(candidates))
    anchor_rows = [row for row in ranking if str(row.get("track_id")) in candidates]
    return {
        "track_id": candidates[0] if len(candidates) == 1 else None,
        "candidate_track_ids": candidates,
        "identity_ambiguous": len(candidates) > 1,
        "goalward_q_m": q_second,
        "X_world_m": float(ranking[1]["goalward_x_m"]),
        "anchor": ranking[1].get("anchor") if len(candidates) == 1 else None,
    }
