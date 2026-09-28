import numpy as np

from ball_localization.evaluation.benchmark_ground_plane_v13 import (
    _in_band,
    _intersect_ball_contact_plane,
)


def test_ground_plane_intersection_uses_ball_radius_and_pitch_bounds():
    class Camera:
        status = "VALID"
        pitch = {"length_m": 105.0, "width_m": 68.0}

        def intersect_z_plane(self, uv, z):
            assert abs(float(z) - 0.11) < 1e-12
            return np.asarray([[10.0, 5.0, z]], dtype=float)

    xyz, status = _intersect_ball_contact_plane(
        Camera(), np.asarray([100.0, 200.0]), ball_radius_m=0.11, pitch_margin_m=6.0
    )
    assert status == "VALID"
    assert xyz == [10.0, 5.0, 0.11]


def test_ground_plane_height_band_boundaries_are_explicit():
    assert _in_band(0.05, 0.0, 0.05)
    assert not _in_band(0.050001, 0.0, 0.05)
    assert _in_band(0.050001, 0.05, 0.10)
    assert _in_band(0.10, 0.05, 0.10)
    assert not _in_band(0.10, 0.10, 0.20)
    assert _in_band(0.200001, 0.20, None)
