"""Stage 4 world-grounded player-pose reconstruction and retained research baselines."""

from .schemas import Stage4Config
from .processor import run_stage4 as run_stage4_legacy3d, preflight_stage4 as preflight_stage4_legacy3d
from .backends.field_converter import FieldConverterV04Config, preflight_v04, run_v04
from .backends.fixed_height_v03 import run_fixed_height_v03
from .backends.sam3d_pitch_refined import Sam3DPitchRefinedConfig, preflight_v05, run_v05

__version__ = "0.5.2"

__all__ = [
    "Stage4Config", "run_stage4_legacy3d", "preflight_stage4_legacy3d",
    "FieldConverterV04Config", "preflight_v04", "run_v04",
    "Sam3DPitchRefinedConfig", "preflight_v05", "run_v05",
    "run_fixed_height_v03",
]
