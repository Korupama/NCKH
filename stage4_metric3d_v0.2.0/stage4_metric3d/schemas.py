from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Stage4Config:
    """Engineering configuration for hybrid metric reconstruction.

    Stage 4 v0.1 uses a hard calibrated-camera ray parameterization. Stage-3
    image points therefore remain the 2D source of truth and are never silently
    moved by the 3D initializer.
    """

    window_radius_frames: int = 13
    min_temporal_frames: int = 7

    nominal_height_m: float = 1.80
    height_sigma_m: float = 0.15
    height_min_m: float = 1.45
    height_max_m: float = 2.15

    depth_beta_init_m: float = 0.25
    depth_beta_min_m: float = -1.20
    depth_beta_max_m: float = 1.20
    depth_prior_sigma_m: float = 0.16
    depth_prior_weight: float = 1.0

    bone_prior_weight: float = 1.0
    ground_weight: float = 1.0
    nonpenetration_weight: float = 1.0
    temporal_weight: float = 1.0
    pitch_bound_weight: float = 0.25

    contact_likelihood_threshold: float = 0.58
    both_contact_y_tolerance_ratio: float = 0.035
    max_temporal_fill_gap_frames: int = 2

    lambda_min_m: float = 1.0
    lambda_max_m: float = 250.0
    pitch_margin_m: float = 5.0

    optimizer_loss: str = "soft_l1"
    optimizer_f_scale: float = 1.0
    optimizer_max_nfev: int = 250
    sparse_jacobian: bool = True
    optimizer_verbose: int = 0
    geometry_max_bone_error_m: float = 0.20
    geometry_max_ground_residual_m: float = 0.20
    geometry_max_negative_z_fraction: float = 0.08

    uncertainty_samples: int = 0
    uncertainty_seed: int = 12345
    uncertainty_mode: str = "selected_frame"
    uncertainty_optimizer_max_nfev: int = 180
    uncertainty_min_usable_fraction: float = 0.80
    base_keypoint_sigma_px: float = 1.5
    max_keypoint_sigma_px: float = 5.0

    allow_degraded_camera: bool = True
    require_initializer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
