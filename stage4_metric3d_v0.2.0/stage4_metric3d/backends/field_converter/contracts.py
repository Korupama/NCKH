from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


STAGE4_V04_VERSION = "stage4-world-grounded-pose-0.4.0"
OUTPUT_SCHEMA = "world-grounded-pose-state-1.0"
HANDOFF_SCHEMA = "stage4-downstream-handoff-2.0"
SAM3D_CACHE_SCHEMA = "stage4-sam3d-cache-1.0"
SAM3D_MANIFEST_SCHEMA = "stage4-sam3d-worker-manifest-1.0"
FC_EXPORT_SCHEMA = "stage4-field-converter-export-1.0"
FC_EXPECTED_JOINTS = 25
FC_TARGET_FPS = 50.0

# Field Converter README training camera statistics.
FC_TRAIN_CAMERA_CENTER_MEAN = (0.112, -75.181, 16.459)
FC_TRAIN_CAMERA_CENTER_STD = (0.120, 6.200, 2.102)
FC_TRAIN_CAMERA_CENTER_MIN = (-0.128, -88.155, 11.765)
FC_TRAIN_CAMERA_CENTER_MAX = (0.323, -66.729, 19.039)
FC_AUTO_OUTLIER_Z_THRESHOLD = 5.0
FC_AUTO_MIN_SCORE_IMPROVEMENT = 0.25


@dataclass(frozen=True)
class FieldConverterV04Config:
    sequence_name: str = "stage4_sequence"
    include_roles: Tuple[str, ...] = ("player", "goalkeeper")
    target_fps: float = FC_TARGET_FPS
    world_alignment: str = "auto"
    sam3d_sign: int = 1
    box_normalization_min_size_px: float = 10.0
    field_converter_device: str = "auto"
    field_converter_batch_size: Optional[int] = None
    save_field_converter_intermediate: bool = True

    # Engineering-only camera compatibility gate. This is not a research
    # accuracy threshold; it detects omitted distortion/convention mismatch.
    camera_projection_compatibility_p95_px: float = 2.0
    camera_projection_compatibility_fail_px: float = 5.0

    # Catastrophic geometry sanity gate, not a publication accuracy claim.
    catastrophic_xy_span_m: float = 4.0
    catastrophic_z_span_m: float = 4.0

    def validate(self) -> None:
        if not self.sequence_name:
            raise ValueError("sequence_name must be non-empty")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        if self.sam3d_sign not in {-1, 1}:
            raise ValueError("sam3d_sign must be -1 or +1")
        if self.world_alignment not in {
            "auto", "none", "rotate_x_180", "rotate_y_180", "rotate_z_180"
        }:
            raise ValueError(f"Unsupported world_alignment: {self.world_alignment}")
        if self.field_converter_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("field_converter_device must be auto/cpu/cuda")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldConverterBundle:
    repo: Optional[Path]
    python_exe: Path
    config_path: Path
    checkpoint_path: Path
    normalization_stats_path: Path
    pitch_points_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": None if self.repo is None else str(self.repo),
            "python_exe": str(self.python_exe),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "normalization_stats_path": str(self.normalization_stats_path),
            "pitch_points_path": str(self.pitch_points_path),
        }
