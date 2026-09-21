from stage4_metric3d.schemas import Stage4Config

def test_defaults_do_not_force_initializer_or_uncertainty():
    cfg = Stage4Config()
    assert cfg.require_initializer is False
    assert cfg.uncertainty_samples == 0
