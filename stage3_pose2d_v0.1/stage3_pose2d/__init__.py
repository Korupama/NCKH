"""Stage 3: tracked 2D WholeBody pose normalization and quality assurance."""

from .processor import Stage3Config, run_stage3
from .stage2_adapter import Stage2Bundle, load_stage2_bundle
from .wholebody133 import WHOLEBODY_KEYPOINT_NAMES

__all__ = [
    "Stage3Config",
    "Stage2Bundle",
    "WHOLEBODY_KEYPOINT_NAMES",
    "load_stage2_bundle",
    "run_stage3",
]

__version__ = "0.1.0"
