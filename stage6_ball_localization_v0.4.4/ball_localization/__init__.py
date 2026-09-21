"""Stage 6 ball detection/tracking/localization."""
from .contracts import BallCandidate2D, BallFrameState, BallTrajectoryState
from .geometry import TemporalRefinementConfig
from .pipeline import refine_existing_state, run_stage6
from .version import PACKAGE_VERSION, STAGE6_VERSION

__all__ = [
    "BallCandidate2D",
    "BallFrameState",
    "BallTrajectoryState",
    "TemporalRefinementConfig",
    "run_stage6",
    "refine_existing_state",
]
__version__ = PACKAGE_VERSION
