from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

POSE_STATUS = ("VALID", "DEGRADED", "REJECTED", "MISSING")
KEYPOINT_STATES = (
    "VALID", "LOW_MODEL_EVIDENCE", "GEOMETRIC_OUTLIER", "TEMPORAL_OUTLIER",
    "LEFT_RIGHT_SUSPECT", "MISSING", "TEMPORAL_IMPUTED",
)

@dataclass
class Stage3Config:
    bbox_margin_for_qa: float = 0.25
    low_score_ratio_to_median: float = 0.35
    min_body_completeness_valid: float = 0.80
    min_body_completeness_reject: float = 0.45
    min_core_completeness_valid: float = 0.80
    min_core_completeness_reject: float = 0.40
    min_feet_completeness_valid: float = 0.50
    min_inside_fraction_valid: float = 0.75
    temporal_accel_threshold: float = 0.35
    temporal_swap_ratio: float = 0.65
    temporal_swap_min_normal_cost: float = 0.12
    emit_temporal_estimates: bool = False
    enable_fallback_reinference: bool = False
    fallback_crop_scales: List[float] = field(default_factory=lambda: [1.0, 1.10, 1.20])
    rtmw_model: Optional[str] = None
    rtmw_device: str = "cpu"
    rtmw_input_width: int = 288
    rtmw_input_height: int = 384

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
