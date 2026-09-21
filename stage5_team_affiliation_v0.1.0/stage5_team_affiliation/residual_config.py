"""Research defaults, not tuned on GSR VALID. Legacy Stage5Config is unchanged."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ResidualConfig:
    core_keep_fraction: float = 0.80
    min_core_tracks: int = 2
    trim_iterations: int = 5
    radius_floor: float = 0.08
    radius_separation_fraction: float = 0.45
    min_centroid_separation: float = 0.20
    min_team_margin: float = 0.08
    min_observations: int = 5
    goal_distance_m: float = 18.0
    min_top2_rate: float = 0.60
    min_pairwise_rate: float = 0.70
    pairwise_ordering_required: bool = False
    ordering_margin_m: float = 0.50
    min_rank_humans: int = 3
    defensive_tail_k: int = 2
    min_tail_separation_m: float = 2.0
    min_tail_vote_rate: float = 0.70
    # v0.2.1 safety switches.  These two decisions were not calibrated on TRAIN
    # in v0.2.0, so they remain diagnostic/abstaining unless a frozen TRAIN
    # configuration explicitly opts in.
    defensive_tail_assignment_enabled: bool = False
    defended_half_assignment_enabled: bool = False
    referee_recovery_enabled: bool = False
    residual_appearance_recovery_enabled: bool = False
    residual_player_recovery_enabled: bool = False
    residual_player_referee_veto_enabled: bool = False
    # v0.3.0 appearance descriptor.  Disabled by default so historical runs
    # remain reproducible; enable only in an explicit source benchmark.
    multiregion_appearance_enabled: bool = False
    multiregion_torso_weight: float = 0.70
    multiregion_lower_weight: float = 0.30
    multiregion_require_lower: bool = True
    residual_referee_appearance_recovery_enabled: bool = False
    residual_player_min_margin: float = 1.0
    residual_player_max_distance: float = 2.0
    residual_player_veto_min_goal_distance_m: float = 22.0
    residual_player_veto_min_top2_rate: float = 0.40
    residual_referee_max_margin: float = 0.0
    residual_referee_min_distance: float = 0.0
    referee_min_goal_distance_m: float = 22.0
    referee_max_top2_rate: float = 0.20
    min_defended_half_separation_m: float = 2.0
    min_defended_half_vote_rate: float = 0.70
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    pitch_margin_m: float = 3.0

    def validate(self):
        for key in ('pairwise_ordering_required', 'defensive_tail_assignment_enabled',
                    'defended_half_assignment_enabled', 'referee_recovery_enabled',
                    'residual_appearance_recovery_enabled',
                    'residual_player_recovery_enabled',
                    'residual_player_referee_veto_enabled',
                    'multiregion_appearance_enabled',
                    'multiregion_require_lower',
                    'residual_referee_appearance_recovery_enabled'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f'{key} must be boolean')
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('Configuration must be finite')
        for key in ('min_core_tracks', 'trim_iterations', 'min_observations', 'min_rank_humans', 'defensive_tail_k'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.min_core_tracks < 2 or self.min_rank_humans < 3:
            raise ValueError('Need at least two core tracks and three humans for ranking')
        if self.defensive_tail_assignment_enabled and self.defended_half_assignment_enabled:
            raise ValueError('Enable only one goalkeeper-team assignment policy')
        for key in ('core_keep_fraction', 'radius_separation_fraction', 'min_team_margin', 'min_top2_rate',
                    'min_pairwise_rate', 'min_tail_vote_rate', 'min_defended_half_vote_rate',
                    'referee_max_top2_rate',
                    'residual_player_veto_min_top2_rate',
                    'residual_player_min_margin'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in (0, 1]')
        if not 0 <= self.residual_referee_max_margin < 1:
            raise ValueError('residual_referee_max_margin must be in [0, 1)')
        if not 0 < self.multiregion_torso_weight <= 1:
            raise ValueError('multiregion_torso_weight must be in (0, 1]')
        if not 0 < self.multiregion_lower_weight <= 1:
            raise ValueError('multiregion_lower_weight must be in (0, 1]')
        if not math.isclose(
                self.multiregion_torso_weight + self.multiregion_lower_weight,
                1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('Multiregion weights must sum to 1')
        if self.residual_referee_max_margin >= self.residual_player_min_margin:
            raise ValueError('Referee/player residual margin bands must not overlap')
        for key in ('radius_floor', 'min_centroid_separation', 'goal_distance_m', 'ordering_margin_m',
                    'min_tail_separation_m', 'min_defended_half_separation_m',
                    'residual_player_veto_min_goal_distance_m',
                    'pitch_length_m', 'pitch_width_m'):
            if getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive')
        if self.pitch_margin_m < 0 or self.referee_min_goal_distance_m <= self.goal_distance_m:
            raise ValueError('Invalid pitch margin/referee distance')
        if not 0 <= self.residual_referee_min_distance <= 2 or not 0 < self.residual_player_max_distance <= 2:
            raise ValueError('Residual appearance distances must lie in [0, 2]')

    def to_dict(self):
        return asdict(self)
