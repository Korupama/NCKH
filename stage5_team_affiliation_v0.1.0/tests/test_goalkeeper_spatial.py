import pytest
from stage5_team_affiliation.config import Stage5Config
from stage5_team_affiliation.goalkeeper import assign_goalkeeper, assign_goalkeeper_spatial, assign_by_defensive_tail
from stage5_team_affiliation.goalkeeper_team_v024 import assign_by_defended_half


@pytest.mark.parametrize("mode", ["spatial_pitch", "spatial_image", "spatial_hybrid", "color_lower_body"])
def test_fixed_defending_policy_overrides_old_modes(mode):
    cfg = Stage5Config(goalkeeper_assignment_mode=mode)
    for x in [-50, 0, 50]:
        result = assign_goalkeeper_spatial(config=cfg, gk_pitch_x=[x], team_pitch_x={0: [-20], 1: [20]})
        assert result['team_id'] is None
        assert result['team_side'] == 'DEFENDING'
        assert result['method'] == 'GOALKEEPER_ALWAYS_DEFENDING'
    result = assign_goalkeeper(None, {}, cfg)
    assert result['team_side'] == 'DEFENDING'


@pytest.mark.parametrize("assign", [assign_by_defensive_tail, assign_by_defended_half])
def test_temporal_assignment_also_uses_fixed_policy(assign):
    result = assign('g', 1, {}, {}, None)
    assert result['team_id'] is None
    assert result['team_status'] == 'PENDING_CONTEXT'
    assert result['team_side'] == 'DEFENDING'
