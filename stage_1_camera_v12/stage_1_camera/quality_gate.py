from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .contracts import CameraState, CameraStatus
from .capabilities import annotate_capabilities
from .policies import GroundGeometryPolicyConfig, Vertical3DPolicyConfig


@dataclass
class QualityGateConfig:
    # Camera-matrix plausibility.
    min_focal_px: float = 100.0
    max_focal_px_factor: float = 20.0
    max_abs_camera_xy_m: float = 250.0
    min_camera_height_m: float = 0.5
    max_camera_height_m: float = 100.0
    max_rotation_orthogonality_error: float = 1e-3
    max_rotation_det_error: float = 1e-3

    # PnLCalib evidence gate. These two reprojection values are traceable to
    # upstream behaviour: heuristic_voting() prefers full/no-RANSAC <= 5 px,
    # while the official SN23 runner rejects results above 38 px.
    preferred_max_rep_err_px: float = 5.0
    hard_max_rep_err_px: float = 38.0
    min_keypoints_for_valid: int = 4
    require_evidence_for_valid: bool = True
    prefer_full_no_ransac_for_valid: bool = True

    # Frozen engineering coverage safety floors. These are NOT literature
    # claims. Full SoccerNet Calibration-2023 VALID review showed that these
    # floors are not the dominant rejection bottleneck, so v12 retains them
    # without overfitting stricter hull/x/y thresholds.
    require_spatial_coverage_for_valid: bool = True
    min_image_hull_ratio_for_valid: float = 0.01
    min_image_x_span_ratio_for_valid: float = 0.20
    min_image_y_span_ratio_for_valid: float = 0.06
    min_image_quadrants_for_valid: int = 2


class CameraQualityGate:
    """Separate critical failures, weak-camera evidence, and optional warnings.

    Status semantics:
    - INVALID: physically/algebraically invalid or above the hard reprojection bound.
    - DEGRADED: solved camera but primary evidence is weak (high residual, fallback
      solution, too few or too spatially concentrated keypoints).
    - VALID: primary evidence is strong enough. Optional evidence may still be
      absent; such cases remain VALID with explicit warnings.

    In particular, requesting PnL line refinement while detecting zero usable
    lines is a WARNING, not an automatic degradation, because a strong keypoint
    camera can still be a valid solution.  The diagnostics record whether line
    refinement was requested and whether it could actually be effective.
    """

    def __init__(
        self,
        cfg: Optional[QualityGateConfig] = None,
        ground_policy: Optional[GroundGeometryPolicyConfig] = None,
        vertical_policy: Optional[Vertical3DPolicyConfig] = None,
    ):
        self.cfg = cfg or QualityGateConfig()
        self.ground_policy = ground_policy or GroundGeometryPolicyConfig(
            preferred_max_rep_err_px=self.cfg.preferred_max_rep_err_px,
            hard_max_rep_err_px=self.cfg.hard_max_rep_err_px,
            min_keypoints=self.cfg.min_keypoints_for_valid,
            min_image_hull_ratio=self.cfg.min_image_hull_ratio_for_valid,
            min_image_x_span_ratio=self.cfg.min_image_x_span_ratio_for_valid,
            min_image_y_span_ratio=self.cfg.min_image_y_span_ratio_for_valid,
            min_image_quadrants=self.cfg.min_image_quadrants_for_valid,
        )
        self.vertical_policy = vertical_policy or Vertical3DPolicyConfig()

    def evaluate(self, cam: CameraState) -> CameraStatus:
        invalid, degraded, warnings = [], [], []
        fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        fmax = self.cfg.max_focal_px_factor * max(cam.image_width, cam.image_height)
        if not (self.cfg.min_focal_px <= fx <= fmax and self.cfg.min_focal_px <= fy <= fmax):
            invalid.append("implausible focal length")

        R = cam.R_world_to_camera
        ortho = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        deterr = float(abs(np.linalg.det(R) - 1.0))
        if ortho > self.cfg.max_rotation_orthogonality_error or deterr > self.cfg.max_rotation_det_error:
            invalid.append("invalid rotation matrix")

        C = cam.camera_center_world_m
        if max(abs(float(C[0])), abs(float(C[1]))) > self.cfg.max_abs_camera_xy_m:
            invalid.append("camera centre too far from pitch")
        if not (self.cfg.min_camera_height_m <= float(C[2]) <= self.cfg.max_camera_height_m):
            invalid.append("implausible camera height in canonical Z-up world")

        line_refinement = {
            "requested": bool(cam.source.get("pnl_refine", False)),
            "effective": False,
            "line_evidence_count": None,
            "reason": "not_applicable",
        }

        if cam.source.get("backend") == "pnlcalib" and cam.diagnostics.get("solver_status") == "SOLVED":
            ev = cam.evidence
            rep = _as_float(ev.get("rep_err_px"))
            n_kp = _as_int(ev.get("num_keypoints_used"))
            n_lines = _as_int(ev.get("num_lines_used"))
            mode = ev.get("mode")
            use_ransac = ev.get("use_ransac")
            refine = bool(cam.source.get("pnl_refine", False))

            if rep is None:
                degraded.append("missing PnLCalib reprojection error")
            elif rep > self.cfg.hard_max_rep_err_px:
                invalid.append(
                    f"PnLCalib rep_err {rep:.3f}px exceeds hard bound {self.cfg.hard_max_rep_err_px:.1f}px"
                )
            elif rep > self.cfg.preferred_max_rep_err_px:
                degraded.append(
                    f"PnLCalib rep_err {rep:.3f}px exceeds preferred {self.cfg.preferred_max_rep_err_px:.1f}px"
                )

            if n_kp is None:
                if self.cfg.require_evidence_for_valid:
                    degraded.append("missing keypoint evidence count")
            elif n_kp < self.cfg.min_keypoints_for_valid:
                degraded.append(
                    f"only {n_kp} keypoints used (<{self.cfg.min_keypoints_for_valid})"
                )

            line_refinement = {
                "requested": refine,
                "effective": bool(refine and n_lines is not None and n_lines > 0),
                "line_evidence_count": n_lines,
                "reason": "line_evidence_available" if refine and n_lines and n_lines > 0 else (
                    "no_line_evidence_above_threshold" if refine and n_lines == 0 else (
                        "line_evidence_count_missing" if refine and n_lines is None else "not_requested"
                    )
                ),
            }
            if refine and n_lines is None:
                warnings.append("PnL refinement requested but line evidence count is missing")
            elif refine and n_lines < 1:
                warnings.append("PnL refinement requested but no line evidence is available")

            if self.cfg.prefer_full_no_ransac_for_valid:
                if mode is None or use_ransac is None:
                    if self.cfg.require_evidence_for_valid:
                        degraded.append("missing PnLCalib solver mode/RANSAC provenance")
                elif not (str(mode) == "full" and int(use_ransac) == 0):
                    degraded.append(f"fallback PnLCalib solution mode={mode}, use_ransac={use_ransac}")

            # Image-space evidence coverage. Only the primary dimensions drive
            # status; world spans are exported as diagnostics for later empirical
            # thresholding and do not currently gate VALID/DEGRADED.
            hull = _as_float(ev.get("keypoint_image_convex_hull_area_ratio"))
            xspan = _as_float(ev.get("keypoint_image_x_span_ratio"))
            yspan = _as_float(ev.get("keypoint_image_y_span_ratio"))
            quadrants = _as_int(ev.get("keypoint_image_quadrants_occupied"))

            coverage_missing = any(v is None for v in (hull, xspan, yspan, quadrants))
            if coverage_missing:
                if self.cfg.require_spatial_coverage_for_valid:
                    degraded.append("missing keypoint spatial-coverage metrics")
                else:
                    warnings.append("keypoint spatial-coverage metrics are unavailable")
            else:
                if hull < self.cfg.min_image_hull_ratio_for_valid:
                    degraded.append(
                        f"keypoint image hull ratio {hull:.4f} is below frozen safety minimum "
                        f"{self.cfg.min_image_hull_ratio_for_valid:.4f}"
                    )
                if xspan < self.cfg.min_image_x_span_ratio_for_valid:
                    degraded.append(
                        f"keypoint image x-span ratio {xspan:.3f} is below frozen safety minimum "
                        f"{self.cfg.min_image_x_span_ratio_for_valid:.3f}"
                    )
                if yspan < self.cfg.min_image_y_span_ratio_for_valid:
                    degraded.append(
                        f"keypoint image y-span ratio {yspan:.3f} is below frozen safety minimum "
                        f"{self.cfg.min_image_y_span_ratio_for_valid:.3f}"
                    )
                if quadrants < self.cfg.min_image_quadrants_for_valid:
                    degraded.append(
                        f"keypoints occupy only {quadrants} image quadrant(s) "
                        f"(<{self.cfg.min_image_quadrants_for_valid})"
                    )

        elif cam.source.get("backend") == "pnlcalib":
            invalid.append("PnLCalib solver did not produce a solved camera")

        cam.diagnostics["line_refinement"] = line_refinement
        cam.diagnostics["quality_gate"] = {
            "invalid_reasons": invalid,
            "degraded_reasons": degraded,
            "warnings": warnings,
            "rotation_orthogonality_error": ortho,
            "rotation_det_error": deterr,
            "config": {
                "preferred_max_rep_err_px": self.cfg.preferred_max_rep_err_px,
                "hard_max_rep_err_px": self.cfg.hard_max_rep_err_px,
                "min_keypoints_for_valid": self.cfg.min_keypoints_for_valid,
                "prefer_full_no_ransac_for_valid": self.cfg.prefer_full_no_ransac_for_valid,
                "require_spatial_coverage_for_valid": self.cfg.require_spatial_coverage_for_valid,
                "min_image_hull_ratio_for_valid": self.cfg.min_image_hull_ratio_for_valid,
                "min_image_x_span_ratio_for_valid": self.cfg.min_image_x_span_ratio_for_valid,
                "min_image_y_span_ratio_for_valid": self.cfg.min_image_y_span_ratio_for_valid,
                "min_image_quadrants_for_valid": self.cfg.min_image_quadrants_for_valid,
                "coverage_thresholds_provisional": False,
                "thresholds_frozen": True,
                "threshold_source": "SoccerNet Calibration-2023 valid review + upstream PnLCalib bounds",
            },
        }

        cam.status = CameraStatus.INVALID if invalid else (CameraStatus.DEGRADED if degraded else CameraStatus.VALID)
        annotate_capabilities(
            cam,
            ground_policy=self.ground_policy,
            vertical_policy=self.vertical_policy,
        )
        return cam.status


def _as_float(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _as_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
