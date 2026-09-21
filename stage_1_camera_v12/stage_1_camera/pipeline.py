from __future__ import annotations

from typing import Any, Dict, Optional

from .contracts import CameraState, PitchSpec
from .pnlcalib_adapter import PnLCalibAdapter
from .quality_gate import CameraQualityGate, QualityGateConfig
from .policies import GroundGeometryPolicyConfig, Vertical3DPolicyConfig


def build_camera_state_from_pnlcalib(
    pnl_result: Dict[str, Any],
    *,
    frame_index: int,
    image_width: int,
    image_height: int,
    pitch: Optional[PitchSpec] = None,
    pnl_refine: Optional[bool] = None,
    source_extra: Optional[Dict[str, Any]] = None,
    quality_cfg: Optional[QualityGateConfig] = None,
    ground_policy: Optional[GroundGeometryPolicyConfig] = None,
    vertical_policy: Optional[Vertical3DPolicyConfig] = None,
) -> CameraState:
    """Normalize a PnLCalib solver result and immediately apply Stage-1 QA."""
    cam = PnLCalibAdapter.camera_state_from_result(
        pnl_result,
        frame_index=frame_index,
        image_width=image_width,
        image_height=image_height,
        pitch=pitch,
        pnl_refine=pnl_refine,
        source_extra=source_extra,
    )
    CameraQualityGate(quality_cfg, ground_policy=ground_policy, vertical_policy=vertical_policy).evaluate(cam)
    return cam
