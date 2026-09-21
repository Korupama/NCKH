from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from .contracts import CameraState, CameraStatus
from .policies import GroundGeometryPolicyConfig, Vertical3DPolicyConfig


def annotate_capabilities(
    cam: CameraState,
    ground_policy: Optional[GroundGeometryPolicyConfig] = None,
    vertical_policy: Optional[Vertical3DPolicyConfig] = None,
) -> Dict[str, Any]:
    """Annotate task-specific camera capabilities with separate frozen policies.

    A camera can be useful for pitch-plane operations while remaining
    insufficient for airborne body geometry.  v12 intentionally prevents the
    single top-level CameraState status from being interpreted as a universal
    3D-readiness flag.
    """
    gp = ground_policy or GroundGeometryPolicyConfig()
    vp = vertical_policy or Vertical3DPolicyConfig()

    ev = cam.evidence or {}
    rep = _float(ev.get("rep_err_px"))
    mode = ev.get("mode")
    ransac = _int(ev.get("use_ransac"))
    n_kp = _int(ev.get("num_keypoints_used"))
    hull = _float(ev.get("keypoint_image_convex_hull_area_ratio"))
    xspan = _float(ev.get("keypoint_image_x_span_ratio"))
    yspan = _float(ev.get("keypoint_image_y_span_ratio"))
    quadrants = _int(ev.get("keypoint_image_quadrants_occupied"))

    q = cam.diagnostics.get("quality_gate", {}) or {}
    q_invalid = list(q.get("invalid_reasons", []) or [])

    ground_reasons = []
    if cam.status == CameraStatus.INVALID or q_invalid:
        ground_status = CameraStatus.INVALID
        ground_reasons.extend(q_invalid or ["camera invalid"])
    else:
        _append_common_evidence_reasons(
            ground_reasons,
            rep=rep,
            n_kp=n_kp,
            hull=hull,
            xspan=xspan,
            yspan=yspan,
            quadrants=quadrants,
            max_rep=gp.preferred_max_rep_err_px,
            min_kp=gp.min_keypoints,
            min_hull=gp.min_image_hull_ratio,
            min_xspan=gp.min_image_x_span_ratio,
            min_yspan=gp.min_image_y_span_ratio,
            min_quadrants=gp.min_image_quadrants,
        )
        if rep is not None and rep > gp.hard_max_rep_err_px:
            ground_status = CameraStatus.INVALID
            ground_reasons.append(
                f"reprojection error {rep:.3f}px exceeds hard bound {gp.hard_max_rep_err_px:.1f}px"
            )
        else:
            ground_status = CameraStatus.DEGRADED if ground_reasons else CameraStatus.VALID

    candidate_diag = cam.diagnostics.get("candidate_diagnostics", {}) or {}
    best_full = (candidate_diag.get("best_by_mode", {}) or {}).get("full")
    full0 = candidate_diag.get("full_no_ransac")

    vertical_reasons = []
    if cam.status == CameraStatus.INVALID or q_invalid:
        vertical_status = CameraStatus.INVALID
        vertical_reasons.extend(q_invalid or ["camera invalid"])
    else:
        if vp.require_full_no_ransac and not (mode == "full" and ransac == 0):
            vertical_reasons.append(
                f"selected solution is {mode}/RANSAC={ransac}, not full/no-RANSAC"
            )
        _append_common_evidence_reasons(
            vertical_reasons,
            rep=rep,
            n_kp=n_kp,
            hull=hull,
            xspan=xspan,
            yspan=yspan,
            quadrants=quadrants,
            max_rep=vp.max_rep_err_px,
            min_kp=vp.min_keypoints,
            min_hull=vp.min_image_hull_ratio,
            min_xspan=vp.min_image_x_span_ratio,
            min_yspan=vp.min_image_y_span_ratio,
            min_quadrants=vp.min_image_quadrants,
        )
        vertical_status = CameraStatus.DEGRADED if vertical_reasons else CameraStatus.VALID

        # Candidate information is diagnostic only; it never promotes the
        # selected direct camera to vertical VALID by itself.
        if vertical_status != CameraStatus.VALID:
            if full0 is not None:
                rep0 = _float(full0.get("rep_err_px"))
                vertical_reasons.append(
                    f"full/no-RANSAC candidate exists at {rep0:.3f}px"
                    if rep0 is not None else "full/no-RANSAC candidate exists"
                )
            elif best_full is not None:
                vertical_reasons.append(
                    f"a full candidate exists only with RANSAC={best_full.get('use_ransac')}"
                )
            elif candidate_diag:
                vertical_reasons.append("candidate sweep found no successful full solution")
            else:
                vertical_reasons.append("candidate diagnostics unavailable")

    temporal_rescue = (cam.temporal or {}).get("rescue", {}) or {}
    if temporal_rescue.get("validated") is True and cam.status != CameraStatus.INVALID:
        vertical_status = CameraStatus.VALID
        vertical_reasons = ["validated bracketed temporal rescue"]

    caps = {
        "ground_geometry": {
            "status": ground_status.value,
            "reasons": _dedupe(ground_reasons),
            "pitch_projection_ready": ground_status != CameraStatus.INVALID,
            "ground_unprojection_ready": ground_status == CameraStatus.VALID,
            "policy_version": gp.policy_version,
        },
        "vertical_3d": {
            "status": vertical_status.value,
            "reasons": _dedupe(vertical_reasons),
            "vertical_geometry_ready": vertical_status == CameraStatus.VALID,
            "source": "temporal_rescue" if temporal_rescue.get("validated") else "direct_single_frame",
            "policy_version": vp.policy_version,
            "best_full_candidate": best_full,
            "full_no_ransac_candidate": full0,
        },
        "offside_3d_ready": bool(vertical_status == CameraStatus.VALID),
        "policies": {
            "ground_geometry": gp.to_dict(),
            "vertical_3d": vp.to_dict(),
            "fallback_ground_solution_does_not_imply_vertical_3d_ready": True,
            "calibration_gate_is_not_offside_accuracy": True,
        },
    }
    cam.diagnostics["capabilities"] = caps
    return caps


def _append_common_evidence_reasons(
    reasons,
    *,
    rep,
    n_kp,
    hull,
    xspan,
    yspan,
    quadrants,
    max_rep,
    min_kp,
    min_hull,
    min_xspan,
    min_yspan,
    min_quadrants,
):
    if rep is None:
        reasons.append("missing reprojection error")
    elif rep > max_rep:
        reasons.append(f"reprojection error {rep:.3f}px exceeds preferred {max_rep:.1f}px")

    if n_kp is None:
        reasons.append("missing keypoint evidence count")
    elif n_kp < min_kp:
        reasons.append(f"only {n_kp} keypoints used (<{min_kp})")

    if hull is None or xspan is None or yspan is None or quadrants is None:
        reasons.append("missing keypoint spatial-coverage metrics")
        return
    if hull < min_hull:
        reasons.append(f"keypoint image hull ratio {hull:.4f} is below {min_hull:.4f}")
    if xspan < min_xspan:
        reasons.append(f"keypoint image x-span ratio {xspan:.3f} is below {min_xspan:.3f}")
    if yspan < min_yspan:
        reasons.append(f"keypoint image y-span ratio {yspan:.3f} is below {min_yspan:.3f}")
    if quadrants < min_quadrants:
        reasons.append(f"keypoints occupy only {quadrants} image quadrant(s) (<{min_quadrants})")


def _dedupe(values):
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _float(v: Optional[Any]):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _int(v: Optional[Any]):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
