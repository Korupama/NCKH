from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Sequence
import numpy as np


def collect_pnlcalib_candidates(
    calib,
    *,
    modes: Sequence[str] = ("full", "ground_plane", "main"),
    ransac_values: Sequence[int] = (0, 5, 10, 15, 25, 50),
    refine: bool = False,
    refine_lines: bool = True,
    selected_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the same PnLCalib candidate grid used by heuristic_voting().

    This is intentionally a thin, duck-typed diagnostic adapter around upstream
    ``FramebyFrameCalib.get_cam_params``.  It does not replace PnLCalib's
    selection policy and does not silently change the selected camera.

    The caller should pass a *fresh* calibration object when possible because
    upstream candidate evaluation mutates calibration state, just as
    ``heuristic_voting`` itself does.
    """
    rows = []
    selected_mode = None if selected_result is None else selected_result.get("mode")
    selected_ransac = None if selected_result is None else selected_result.get("use_ransac")
    selected_rep = _finite_float(None if selected_result is None else selected_result.get("rep_err"))

    for mode in modes:
        for ransac in ransac_values:
            row: Dict[str, Any] = {
                "mode": str(mode),
                "use_ransac": int(ransac),
                "success": False,
                "rep_err_px": None,
                "selected": False,
            }
            try:
                cam_params, rep_err = calib.get_cam_params(
                    mode=mode,
                    use_ransac=ransac,
                    refine=refine,
                    refine_w_lines=refine_lines,
                )
                rep = _finite_float(rep_err)
                if cam_params is not None and rep is not None:
                    row["success"] = True
                    row["rep_err_px"] = rep
                    row["x_focal_length"] = _finite_float(cam_params.get("x_focal_length"))
                    row["y_focal_length"] = _finite_float(cam_params.get("y_focal_length"))
                    pos = cam_params.get("position_meters")
                    if pos is not None:
                        row["position_meters_pnl"] = [float(x) for x in np.asarray(pos).reshape(3)]
            except Exception as exc:  # diagnostic grid must not kill the main inference path
                row["error"] = f"{type(exc).__name__}: {exc}"

            if (
                selected_mode is not None
                and selected_ransac is not None
                and row["mode"] == str(selected_mode)
                and row["use_ransac"] == int(selected_ransac)
                and row["success"]
            ):
                if selected_rep is None or abs(row["rep_err_px"] - selected_rep) <= 1e-3:
                    row["selected"] = True
            rows.append(row)

    successful = [r for r in rows if r["success"]]
    successful_sorted = sorted(successful, key=lambda r: (r["rep_err_px"], r["mode"], r["use_ransac"]))

    best_by_mode: Dict[str, Any] = {}
    for mode in modes:
        mrows = [r for r in successful_sorted if r["mode"] == mode]
        best_by_mode[str(mode)] = _compact(mrows[0]) if mrows else None

    full_no_ransac = next(
        (r for r in successful if r["mode"] == "full" and r["use_ransac"] == 0),
        None,
    )
    best_overall = successful_sorted[0] if successful_sorted else None

    return {
        "grid": {
            "modes": list(modes),
            "ransac_values": [int(x) for x in ransac_values],
            "refine": bool(refine),
            "refine_lines": bool(refine_lines),
        },
        "num_attempted": len(rows),
        "num_successful": len(successful),
        "selected": {
            "mode": selected_mode,
            "use_ransac": selected_ransac,
            "rep_err_px": selected_rep,
        },
        "best_overall": _compact(best_overall),
        "best_by_mode": best_by_mode,
        "full_no_ransac": _compact(full_no_ransac),
        "candidates": rows,
    }


def clone_calibration_for_candidate_diagnostics(calib, factory):
    """Create a fresh PnLCalib calibration object from an already-updated one.

    ``factory`` is typically ``FramebyFrameCalib`` imported in the notebook.
    Pixel coordinates in an already-updated ``denormalize=True`` object are
    already denormalized, so the clone deliberately uses ``denormalize=False``.
    """
    cloned = factory(iwidth=calib.image_width, iheight=calib.image_height, denormalize=False)
    cloned.update(deepcopy(getattr(calib, "keypoints_dict", {})), deepcopy(getattr(calib, "lines_dict", {})))
    return cloned


def _compact(row):
    if row is None:
        return None
    keep = (
        "mode", "use_ransac", "success", "rep_err_px", "selected",
        "x_focal_length", "y_focal_length", "position_meters_pnl",
    )
    return {k: row.get(k) for k in keep if k in row}


def _finite_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None
