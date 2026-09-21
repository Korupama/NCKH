from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class Stage5Config:
    sample_every_n_frames: int = 2
    max_samples_per_track: int = 30
    min_valid_torso_frames: int = 3
    min_valid_lower_body_frames: int = 2
    random_state: int = 23
    kmeans_n_init: int = 20
    crop_padding_px: int = 2
    torso_erode_fraction: float = 0.08
    green_hue_min: int = 32
    green_hue_max: int = 92
    green_min_saturation: int = 45
    min_pixel_saturation: int = 18
    min_pixel_value: int = 24
    max_pixel_value: int = 245
    outlier_mad_factor: float = 4.5
    min_cluster_margin: float = 0.08
    goalkeeper_min_margin: float = 0.10
    goalkeeper_max_distance_ratio: float = 1.35
    allow_bbox_torso_fallback: bool = True
    allow_bbox_lower_body_fallback: bool = True

    def validate(self) -> None:
        if self.sample_every_n_frames < 1:
            raise ValueError("sample_every_n_frames must be >= 1")
        if self.max_samples_per_track < 1:
            raise ValueError("max_samples_per_track must be >= 1")
        if self.min_valid_torso_frames < 1:
            raise ValueError("min_valid_torso_frames must be >= 1")
        if self.min_valid_lower_body_frames < 1:
            raise ValueError("min_valid_lower_body_frames must be >= 1")
        if not (0 <= self.green_hue_min < self.green_hue_max <= 179):
            raise ValueError("green hue range must be in OpenCV HSV hue space [0,179]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
