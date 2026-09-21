from types import SimpleNamespace

import numpy as np

from stage4_metric3d.optimizer import FrameInput
from stage4_metric3d.schemas import Stage4Config
from stage4_metric3d.uncertainty import pixel_sensitivity_monte_carlo


def _frame():
    return FrameInput(
        frame_index=86,
        camera=None,
        uv23=np.column_stack([np.arange(23, dtype=float), np.arange(23, dtype=float) + 10.0]),
        state_weights23=np.ones(23, dtype=float),
        bbox_xyxy=np.asarray([0.0, 0.0, 50.0, 100.0]),
        contact=None,
        initializer=None,
        uv_source23=("STAGE3_RAW",) * 23,
    )


def test_plausible_max_nfev_samples_are_retained_and_deterministic(monkeypatch):
    def fake_solve(frames, config, **kwargs):
        uv = frames[0].uv23
        xyz = np.zeros((1, 23, 3), dtype=float)
        xyz[0, :, 0] = uv[:, 0] * 0.01
        xyz[0, :, 1] = uv[:, 1] * 0.01
        return SimpleNamespace(
            status="MAX_NFEV",
            xyz_world_m=xyz,
            diagnostics={"geometry_quality_gate": {"plausible": True}},
        )

    monkeypatch.setattr("stage4_metric3d.uncertainty.solve_metric_pose", fake_solve)
    cfg = Stage4Config(uncertainty_samples=8, uncertainty_seed=77)
    frozen = SimpleNamespace(estimated_height_m=1.8, depth_beta_m=None, initializer_used=False)
    kwargs = dict(
        selected_frame_index=86,
        states23_by_frame={86: ("VALID",) * 23},
        track_id="track_004",
        mode="selected_frame",
    )
    first = pixel_sensitivity_monte_carlo([_frame()], frozen, cfg, **kwargs)
    second = pixel_sensitivity_monte_carlo([_frame()], frozen, cfg, **kwargs)
    assert first["status"] == "DEGRADED"
    assert first["samples_usable"] == 8
    assert first["sample_acceptance_counts"] == {"DEGRADED_MAX_NFEV": 8}
    assert first["longitudinal"]["attack_direction_applied"] is False
    assert first["longitudinal"]["legal_extrema_x"] == second["longitudinal"]["legal_extrema_x"]


def test_implausible_samples_are_reported_not_silently_dropped(monkeypatch):
    def fake_solve(frames, config, **kwargs):
        return SimpleNamespace(
            status="CONVERGED",
            xyz_world_m=np.zeros((1, 23, 3), dtype=float),
            diagnostics={"geometry_quality_gate": {"plausible": False}},
        )

    monkeypatch.setattr("stage4_metric3d.uncertainty.solve_metric_pose", fake_solve)
    cfg = Stage4Config(uncertainty_samples=3)
    frozen = SimpleNamespace(estimated_height_m=1.8, depth_beta_m=None, initializer_used=False)
    result = pixel_sensitivity_monte_carlo(
        [_frame()], frozen, cfg,
        selected_frame_index=86,
        states23_by_frame={86: ("VALID",) * 23},
        track_id="track_010",
    )
    assert result["status"] == "FAILED"
    assert result["samples_usable"] == 0
    assert result["sample_failure_counts"] == {"IMPLAUSIBLE_GEOMETRY": 3}
