from __future__ import annotations

from typing import Any, Dict, Iterable
import numpy as np

# SoccerNet/sn-calibration defines elevated goal points with negative Z. Stage 6
# uses the frozen canonical pitch frame with +Z upward.  A pure Z reflection
# would make world-to-camera matrices improper (det=-1), so we use the proper
# 180-degree rotation around the pitch X axis.  X, the offside-critical
# goal-to-goal axis, is preserved exactly; Y and Z change sign.
SOCCERNET_TO_STAGE6_WORLD = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def soccernet_xyz_to_stage6(xyz: Iterable[float]) -> np.ndarray:
    """Convert one SoccerNet pitch/world point to Stage-6 canonical XYZ."""
    p = np.asarray(list(xyz), dtype=np.float64).reshape(3)
    return SOCCERNET_TO_STAGE6_WORLD @ p


def stage6_xyz_to_soccernet(xyz: Iterable[float]) -> np.ndarray:
    """Inverse conversion. The transform is an involution (T == T^-1)."""
    p = np.asarray(list(xyz), dtype=np.float64).reshape(3)
    return SOCCERNET_TO_STAGE6_WORLD @ p


def soccernet_rotation_to_stage6(R_world_to_camera: np.ndarray) -> np.ndarray:
    """Convert SoccerNet world->camera rotation to Stage-6 world->camera.

    If X_stage6 = T X_sn and C_stage6 = T C_sn, then
      X_cam = R_sn (X_sn-C_sn) = R_sn T^T (X_stage6-C_stage6).
    T is symmetric here, hence R_stage6 = R_sn T.
    """
    R = np.asarray(R_world_to_camera, dtype=np.float64).reshape(3, 3)
    out = R @ SOCCERNET_TO_STAGE6_WORLD.T
    return out


def coordinate_transform_metadata() -> Dict[str, Any]:
    return {
        "source_frame": "SOCCERNET_CALIBRATION_WORLD",
        "target_frame": "STAGE6_CANONICAL_PITCH_XYZ",
        "matrix_source_to_target": SOCCERNET_TO_STAGE6_WORLD.tolist(),
        "operation": "proper_rotation_180deg_about_X",
        "preserves_X": True,
        "flips_Y": True,
        "flips_Z": True,
        "target_axes": {
            "X": "goal-to-goal",
            "Y": "touchline-to-touchline",
            "Z": "up-positive",
        },
    }
