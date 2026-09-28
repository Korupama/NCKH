__version__ = "0.12.1"

from .contracts import CameraState, CameraStatus, CameraTimeline, Distortion, PitchSpec
from .pnlcalib_adapter import PnLCalibAdapter
from .quality_gate import CameraQualityGate, QualityGateConfig
from .policies import GroundGeometryPolicyConfig, Vertical3DPolicyConfig
from .evidence import (
    summarize_keypoint_coverage,
    build_keypoint_correspondences,
    reprojection_error_from_correspondences,
)
from .capabilities import annotate_capabilities
from .candidates import collect_pnlcalib_candidates, clone_calibration_for_candidate_diagnostics
from .geometry import principal_point_px, centre_ray_pitch_intersection, camera_view_metadata

__all__ = [
    "CameraState", "CameraStatus", "CameraTimeline", "Distortion", "PitchSpec",
    "PnLCalibAdapter", "CameraQualityGate", "QualityGateConfig",
    "GroundGeometryPolicyConfig", "Vertical3DPolicyConfig",
    "summarize_keypoint_coverage", "build_keypoint_correspondences",
    "reprojection_error_from_correspondences", "annotate_capabilities",
    "collect_pnlcalib_candidates", "clone_calibration_for_candidate_diagnostics",
    "principal_point_px", "centre_ray_pitch_intersection", "camera_view_metadata",
    "SoccerNetBenchmarkConfig", "SoccerNetCalibrationBenchmark", "inspect_dataset",
    "review_vertical_policy_csv",
]

from .evaluation.soccernet_benchmark import (
    SoccerNetBenchmarkConfig, SoccerNetCalibrationBenchmark, inspect_dataset,
    review_vertical_policy_csv,
)
