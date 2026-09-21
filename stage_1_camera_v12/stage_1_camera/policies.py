from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GroundGeometryPolicyConfig:
    """Frozen Stage-1 policy for pitch-plane geometry.

    The 5 px preferred and 38 px hard reprojection bounds are traceable to the
    PnLCalib/SoccerNet calibration pipeline.  The image-space coverage values
    are retained safety floors from the pre-v12 implementation; full VALID
    review showed they were not the dominant source of rejection.
    """

    preferred_max_rep_err_px: float = 5.0
    hard_max_rep_err_px: float = 38.0
    min_keypoints: int = 4
    min_image_hull_ratio: float = 0.01
    min_image_x_span_ratio: float = 0.20
    min_image_y_span_ratio: float = 0.06
    min_image_quadrants: int = 2

    policy_version: str = "ground-v1"
    thresholds_frozen: bool = True
    threshold_source: str = (
        "PnLCalib upstream reprojection bounds + Stage-1 engineering spatial "
        "safety floors reviewed on SoccerNet Calibration-2023 valid"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Vertical3DPolicyConfig:
    """Frozen direct single-frame policy for vertical/offside 3D readiness.

    These thresholds were selected *only* to gate camera-calibration evidence,
    not to claim offside-decision accuracy.  They were frozen after analysis of
    the SoccerNet Calibration-2023 VALID split (3212 frames): among otherwise
    plausible direct full/no-RANSAC cameras with <=5 px reprojection error,
    requiring >=8 keypoints and coverage of >=3 image quadrants improved the
    calibration-accuracy distribution while the stricter hull/x/y experiments
    added almost no benefit.  The hull/x/y values below therefore remain basic
    safety floors rather than validation-tuned cut points.
    """

    require_full_no_ransac: bool = True
    max_rep_err_px: float = 5.0
    min_keypoints: int = 8
    min_image_quadrants: int = 3
    min_image_hull_ratio: float = 0.01
    min_image_x_span_ratio: float = 0.20
    min_image_y_span_ratio: float = 0.06

    policy_version: str = "vertical-3d-v1"
    thresholds_frozen: bool = True
    threshold_source: str = "SoccerNet Calibration-2023 valid"
    threshold_source_frames: int = 3212
    threshold_source_report_schema: str = "1.0"

    # Validation-set audit numbers.  These describe the calibration subset that
    # passed this gate; they are provenance, not target guarantees.
    validation_ready_count: int = 1007
    validation_ready_rate: float = 0.3135118306351183
    validation_mean_frame_accuracy: float = 0.969977218676229
    validation_p10_frame_accuracy: float = 0.8888888955116272
    validation_fraction_frame_accuracy_ge_0_9: float = 0.8828202581926514

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
