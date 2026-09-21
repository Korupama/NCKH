__version__ = "1.3.0"

from .contracts import ReplayContext, EntityTrackState, EntityTrack, TrackObservation
from .stage1_adapter import replay_context_from_stage1_workspace, Stage1CompatibilityError
from .frame_window import extract_analysis_window
from .consolidation import consolidate_human_detections, PhysicalHumanHypothesis
