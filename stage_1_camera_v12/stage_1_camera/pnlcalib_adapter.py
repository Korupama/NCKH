from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import numpy as np

from .contracts import CameraState, CameraStatus, Distortion, PitchSpec


class PnLCalibAdapter:
    """Normalize official PnLCalib output into the project's canonical CameraState.

    PnLCalib/SoccerNet uses a pitch-centred coordinate system where elevated goal
    points and camera height are represented with negative Z. The project canonical
    world is right-handed and Z-up. We therefore use the proper rotation
    ``T = diag(1,-1,-1)``, equivalent to a 180-degree rotation about X:

        X_can = T @ X_pnl
        C_can = T @ C_pnl
        R_can = R_pnl @ T

    This preserves image projection while exposing a stable downstream Z-up world.
    """

    PNL_TO_CANONICAL = np.diag([1.0, -1.0, -1.0])

    @staticmethod
    def camera_state_from_result(
        result: Dict[str, Any],
        *,
        frame_index: int,
        image_width: int,
        image_height: int,
        timestamp_sec: Optional[float] = None,
        pitch: Optional[PitchSpec] = None,
        pnl_refine: Optional[bool] = None,
        source_extra: Optional[Dict[str, Any]] = None,
    ) -> CameraState:
        if result is None or "cam_params" not in result:
            return CameraState(
                frame_index=frame_index,
                image_width=image_width,
                image_height=image_height,
                K=np.eye(3),
                R_world_to_camera=np.eye(3),
                camera_center_world_m=np.zeros(3),
                pitch=pitch or PitchSpec(),
                timestamp_sec=timestamp_sec,
                status=CameraStatus.INVALID,
                source={"backend": "pnlcalib", "reason": "missing cam_params", **(source_extra or {})},
                diagnostics={"solver_status": "FAILED"},
            )

        cp = result["cam_params"]
        required = ["x_focal_length", "y_focal_length", "principal_point", "position_meters", "rotation_matrix"]
        missing = [k for k in required if k not in cp]
        if missing:
            raise KeyError(f"PnLCalib cam_params missing {missing}")

        K = np.array(
            [
                [float(cp["x_focal_length"]), 0.0, float(cp["principal_point"][0])],
                [0.0, float(cp["y_focal_length"]), float(cp["principal_point"][1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        R_pnl = np.asarray(cp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        C_pnl = np.asarray(cp["position_meters"], dtype=np.float64).reshape(3)
        T = PnLCalibAdapter.PNL_TO_CANONICAL
        R = R_pnl @ T
        C = T @ C_pnl

        distortion = Distortion(
            radial=list(cp.get("radial_distortion", cp.get("radial_distortion_coefficients", [])) or []),
            tangential=list(cp.get("tangential_distortion", cp.get("tangential_distortion_coefficients", [])) or []),
            thin_prism=list(cp.get("thin_prism_distortion", cp.get("thin_prism_distortion_coefficients", [])) or []),
        )

        raw_evidence = dict(result.get("evidence", {}) or {})
        refine_flag = pnl_refine
        if refine_flag is None:
            refine_flag = result.get("pnl_refine", result.get("refine_lines"))
        if refine_flag is None:
            refine_flag = False

        source = {
            "backend": "pnlcalib",
            "backend_world": "soccernet_pitch_centered_z_down",
            "canonical_world": "pitch_centered_right_handed_z_up",
            "world_transform_pnl_to_canonical": T.tolist(),
            "pnl_refine": bool(refine_flag),
            **(source_extra or {}),
        }

        evidence = {
            "raw_result_keys": sorted(result.keys()),
            "rep_err_px": _maybe_float(result.get("rep_err")),
            "mode": result.get("mode"),
            "use_ransac": result.get("use_ransac"),
            "calib_plane": result.get("calib_plane"),
            "num_keypoints_used": raw_evidence.get("num_keypoints_used", result.get("num_keypoints")),
            "num_lines_used": raw_evidence.get("num_lines_used", result.get("num_lines")),
            "kp_threshold": raw_evidence.get("kp_threshold"),
            "line_threshold": raw_evidence.get("line_threshold"),
        }
        # Keep backend evidence extensible without overwriting normalized keys.
        for key, value in raw_evidence.items():
            evidence.setdefault(key, value)

        return CameraState(
            frame_index=frame_index,
            image_width=image_width,
            image_height=image_height,
            K=K,
            R_world_to_camera=R,
            camera_center_world_m=C,
            distortion=distortion,
            pitch=pitch or PitchSpec(),
            timestamp_sec=timestamp_sec,
            # A solver result exists, but it is not VALID until the evidence-aware
            # quality gate runs. This avoids equating algebraic solvability with
            # calibration trustworthiness.
            status=CameraStatus.DEGRADED,
            source=source,
            evidence=evidence,
            diagnostics={
                "solver_status": "SOLVED",
                "raw_cam_params": cp,
                "pnl_camera_center_world_m": C_pnl.tolist(),
                "pnl_rotation_matrix": R_pnl.tolist(),
                "candidate_diagnostics": result.get("candidate_diagnostics", {}),
            },
        )

    @staticmethod
    def canonical_to_pnl_world(xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        return pts @ PnLCalibAdapter.PNL_TO_CANONICAL.T

    @staticmethod
    def from_json(path: str, **kwargs) -> CameraState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return PnLCalibAdapter.camera_state_from_result(data, **kwargs)


def _maybe_float(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None
