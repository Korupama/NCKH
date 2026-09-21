from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Stage4ProjectionConfig:
    """Configuration for the v0.3.1 quality-gated metric projection path."""

    window_radius_frames: int = 13
    reference_player_height_m: float = 1.80
    height_sensitivity_delta_m: float = 0.10
    pitch_margin_m: float = 5.0
    allow_degraded_camera: bool = True
    height_profile: str = "canonical-body-height-v1"
    export_visualization: bool = True

    # Provisional physical-coherence checks. These are safety limits, not
    # learned accuracy thresholds, and must be frozen on a validation split.
    compactness_warn_axis_span_m: float = 1.25
    compactness_reject_axis_span_m: float = 2.00
    compactness_warn_diameter_m: float = 1.75
    compactness_reject_diameter_m: float = 2.50
    compactness_warn_foot_to_upper_m: float = 0.75
    compactness_reject_foot_to_upper_m: float = 1.50

    ground_anchor_cluster_radius_m: float = 1.00
    ground_anchor_min_inliers: int = 3
    ground_anchor_max_inlier_radius_m: float = 0.75

    def validate(self) -> None:
        if self.window_radius_frames < 0:
            raise ValueError("window_radius_frames must be non-negative")
        if not 1.0 <= self.reference_player_height_m <= 2.5:
            raise ValueError("reference_player_height_m must be in [1.0, 2.5]")
        if not 0.0 < self.height_sensitivity_delta_m < self.reference_player_height_m:
            raise ValueError("height_sensitivity_delta_m must be positive and below reference height")
        if self.pitch_margin_m < 0.0:
            raise ValueError("pitch_margin_m must be non-negative")
        if not 0.0 < self.compactness_warn_axis_span_m < self.compactness_reject_axis_span_m:
            raise ValueError("compactness axis-span thresholds must be positive and ordered")
        if not 0.0 < self.compactness_warn_diameter_m < self.compactness_reject_diameter_m:
            raise ValueError("compactness diameter thresholds must be positive and ordered")
        if not 0.0 < self.compactness_warn_foot_to_upper_m < self.compactness_reject_foot_to_upper_m:
            raise ValueError("foot-to-upper thresholds must be positive and ordered")
        if self.ground_anchor_cluster_radius_m <= 0.0:
            raise ValueError("ground_anchor_cluster_radius_m must be positive")
        if self.ground_anchor_min_inliers < 2:
            raise ValueError("ground_anchor_min_inliers must be at least 2")
        if self.ground_anchor_max_inlier_radius_m <= 0.0:
            raise ValueError("ground_anchor_max_inlier_radius_m must be positive")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)
