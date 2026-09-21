from __future__ import annotations

import numpy as np


def _pct(values: list[float], q: float) -> float | None:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    return None if arr.size == 0 else float(np.percentile(arr, q))


def summarize_observations(observations: list[dict], *, below_pitch_tolerance_m: float) -> dict:
    reproj: list[float] = []
    xspan: list[float] = []
    yspan: list[float] = []
    zspan: list[float] = []
    diameter: list[float] = []
    corrections: list[float] = []
    ground_residual_cm: list[float] = []
    below = 0
    finite_joint_count = 0
    for obs in observations:
        if not obs.get("valid"):
            continue
        for joint in obs.get("joints", []):
            e = joint.get("reprojection_error_px")
            if e is not None and np.isfinite(e):
                reproj.append(float(e))
        pts = np.asarray([j["xyz_world_m"] for j in obs.get("joints", []) if j.get("xyz_world_m") is not None], dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] > 0:
            xspan.append(float(np.ptp(pts[:, 0])))
            yspan.append(float(np.ptp(pts[:, 1])))
            zspan.append(float(np.ptp(pts[:, 2])))
            dif = pts[:, None, :] - pts[None, :, :]
            diameter.append(float(np.sqrt(np.sum(dif * dif, axis=-1)).max()))
            finite_joint_count += int(pts.shape[0])
            below += int(np.sum(pts[:, 2] < -abs(float(below_pitch_tolerance_m))))
        c = obs.get("translation", {}).get("sam_correction_m")
        if c is not None and np.isfinite(c):
            corrections.append(float(c))
        gr = obs.get("quality", {}).get("ground_contact_residual_cm")
        if gr is not None and np.isfinite(gr):
            ground_residual_cm.append(float(gr))
    return {
        "ground_vs_sam_disagreement_m": {key: _pct([o.get("quality", {}).get("ground_vs_sam_disagreement_m") for o in observations if o.get("quality", {}).get("ground_vs_sam_disagreement_m") is not None], pct) for key, pct in (("median",50),("p95",95))},
        "ground_candidate_spread_m": {key: _pct([o.get("quality", {}).get("ground_candidate_spread_m") for o in observations if o.get("quality", {}).get("ground_candidate_spread_m") is not None], pct) for key, pct in (("median",50),("p95",95))},
        "ground_vs_refined_m": {key: _pct([o.get("quality", {}).get("ground_vs_refined_m") for o in observations if o.get("quality", {}).get("ground_vs_refined_m") is not None], pct) for key, pct in (("median",50),("p95",95))},
        "reprojection_error_px": {
            "median": _pct(reproj, 50), "p90": _pct(reproj, 90), "p95": _pct(reproj, 95), "count": len(reproj)
        },
        "skeleton_span_m": {
            "x_median": _pct(xspan, 50), "x_p95": _pct(xspan, 95),
            "y_median": _pct(yspan, 50), "y_p95": _pct(yspan, 95),
            "z_median": _pct(zspan, 50), "z_p95": _pct(zspan, 95),
            "diameter_median": _pct(diameter, 50), "diameter_p95": _pct(diameter, 95),
        },
        "sam_root_correction_m": {"median": _pct(corrections, 50), "p95": _pct(corrections, 95)},
        "ground_contact_residual_cm": {"median": _pct(ground_residual_cm, 50), "p95": _pct(ground_residual_cm, 95)},
        "below_pitch_joint_rate": None if finite_joint_count == 0 else float(below / finite_joint_count),
    }
