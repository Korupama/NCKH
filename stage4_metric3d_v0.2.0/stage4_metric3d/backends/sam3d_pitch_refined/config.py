from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sam3DPitchRefinedConfig:
    window_radius_frames: int = 15
    min_rtmw_joint_weight: float = 0.25
    reprojection_sigma_px: float = 8.0
    sam_prior_sigma_xyz_m: tuple[float, float, float] = (1.5, 1.5, 2.0)
    ground_sigma_xyz_m: tuple[float, float, float] = (0.10, 0.10, 0.10)
    grounded_sam_prior_sigma_m: float = 10.0
    max_ground_refinement_m: float = 0.75
    temporal_second_difference_sigma_m: float = 0.50
    max_ground_prior_distance_m: float = 2.50
    max_translation_correction_m: float = 5.0
    use_temporal: bool = True
    optimizer_loss: str = "soft_l1"
    optimizer_max_nfev: int = 300
    camera_convention_pass_p95_px: float = 2.0
    camera_convention_warn_p95_px: float = 5.0
    catastrophic_xy_span_m: float = 3.0
    catastrophic_z_span_m: float = 3.0
    catastrophic_diameter_m: float = 3.5
    sanity_reprojection_p95_px: float = 50.0
    below_pitch_tolerance_m: float = 0.10

    def validate(self) -> None:
        if self.window_radius_frames < 0:
            raise ValueError("window_radius_frames must be >= 0")
        if not 0 <= self.min_rtmw_joint_weight <= 1:
            raise ValueError("min_rtmw_joint_weight must be in [0,1]")
        for name, value in (
            ("reprojection_sigma_px", self.reprojection_sigma_px),
            ("temporal_second_difference_sigma_m", self.temporal_second_difference_sigma_m),
            ("max_ground_prior_distance_m", self.max_ground_prior_distance_m),
            ("max_translation_correction_m", self.max_translation_correction_m),
            ("grounded_sam_prior_sigma_m", self.grounded_sam_prior_sigma_m),
            ("max_ground_refinement_m", self.max_ground_refinement_m),
        ):
            if float(value) <= 0:
                raise ValueError(f"{name} must be > 0")
        if any(float(x) <= 0 for x in self.sam_prior_sigma_xyz_m):
            raise ValueError("sam_prior_sigma_xyz_m must be positive")
        if any(float(x) <= 0 for x in self.ground_sigma_xyz_m):
            raise ValueError("ground_sigma_xyz_m must be positive")
