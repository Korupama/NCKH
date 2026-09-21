import math

import numpy as np

from ball_localization.camera import CameraStateLite
from ball_localization.contracts import BallCandidate2D
from ball_localization.geometry import (
    TemporalRefinementConfig,
    angular_radius_from_bbox,
    estimate_frame,
    refine_diameter_series,
    refine_temporal_trajectory,
)


def _candidate(fi: int, center, diameter: float, score: float = 0.9) -> BallCandidate2D:
    cx, cy = map(float, center)
    h = float(diameter) / 2.0
    return BallCandidate2D(
        fi,
        f"c{fi}",
        [cx - h, cy - h, cx + h, cy + h],
        [cx, cy],
        score,
        "synthetic",
        float(diameter),
    )


def _look_at_camera(frame_index: int = 0) -> CameraStateLite:
    C = np.asarray([0.0, -60.0, 12.0], dtype=float)
    target = np.asarray([28.0, 2.0, 0.0], dtype=float)
    forward = target - C
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)
    K = np.asarray([[1600.0, 0.0, 960.0], [0.0, 1600.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
    return CameraStateLite(
        frame_index,
        1920,
        1080,
        K,
        R,
        C,
        np.zeros(12, dtype=float),
        "VALID",
        {"length_m": 105.0, "width_m": 68.0},
        frame_index / 30.0,
    )


def _ideal_diameter(camera: CameraStateLite, xyz: np.ndarray, radius: float = 0.11) -> tuple[list[float], float]:
    center = camera.project_world(xyz)[0]
    rho = float(np.linalg.norm(xyz - camera.camera_center_world_m))
    target_alpha = math.asin(radius / rho)
    lo, hi = 0.05, 100.0
    for _ in range(60):
        d = (lo + hi) / 2.0
        bbox = [center[0] - d / 2, center[1] - d / 2, center[0] + d / 2, center[1] + d / 2]
        alpha = angular_radius_from_bbox(camera, bbox)["alpha_rad"]
        if alpha < target_alpha:
            lo = d
        else:
            hi = d
    return center.astype(float).tolist(), (lo + hi) / 2.0


def test_log_diameter_refiner_suppresses_spike_and_short_gap():
    fis = list(range(21))
    truth = np.asarray([10.0 + 0.08 * i for i in fis], dtype=float)
    noise = np.asarray([0.00, .02, -.03, .01, .00, .04, -.02, .01, .00, .25, -.01, .02, -.04, .00, .01, -.03, .02, .00, .03, -.02, .00])
    observed = truth * np.exp(noise)
    selected = {fi: _candidate(fi, [100 + fi, 100], observed[fi]) for fi in fis}
    selected[7] = None
    selected[8] = None
    cfg = TemporalRefinementConfig(diameter_second_diff_weight=10.0, max_gap_frames=3)
    refined = refine_diameter_series(fis, selected, config=cfg)
    ref = np.asarray([refined[fi]["refined_diameter_px"] for fi in fis], dtype=float)
    direct_mask = np.asarray([selected[fi] is not None for fi in fis])
    raw_rmse = float(np.sqrt(np.mean((observed[direct_mask] - truth[direct_mask]) ** 2)))
    refined_rmse = float(np.sqrt(np.mean((ref[direct_mask] - truth[direct_mask]) ** 2)))
    assert refined_rmse < raw_rmse * 0.55
    assert refined[7]["diameter_imputed"] is True
    assert refined[8]["diameter_imputed"] is True
    assert abs(ref[7] - truth[7]) / truth[7] < 0.05


def test_temporal_3d_reduces_noisy_size_prior_error_and_bridges_short_gap():
    rng = np.random.default_rng(1234)
    fis = list(range(31))
    cameras = {fi: _look_at_camera(fi) for fi in fis}
    truth = {fi: np.asarray([24.0 + 0.18 * fi, 2.0, 0.11], dtype=float) for fi in fis}
    selected = {}
    raw_errors = []
    for fi in fis:
        center, d = _ideal_diameter(cameras[fi], truth[fi])
        log_noise = float(rng.normal(0.0, 0.075))
        if fi in {6, 19, 24}:
            log_noise += 0.24
        noisy_d = d * math.exp(log_noise)
        cand = _candidate(fi, center, noisy_d)
        selected[fi] = cand
        state = estimate_frame(cameras[fi], cand, fps=30.0, mode="size-prior", max_size_prior_height_m=100.0)
        if state.selected_center_xyz_world_m is not None:
            raw_errors.append(float(np.linalg.norm(np.asarray(state.selected_center_xyz_world_m) - truth[fi])))
    selected[13] = None
    selected[14] = None

    cfg = TemporalRefinementConfig(
        diameter_second_diff_weight=9.0,
        max_gap_frames=3,
        range_sigma_fraction=0.08,
        position_second_diff_sigma_m=0.60,
        position_third_diff_sigma_m=0.40,
        pitch_margin_m=8.0,
        max_height_m=8.0,
    )
    result = refine_temporal_trajectory(
        cameras_by_frame=cameras,
        selected=selected,
        frame_indices=fis,
        fps=30.0,
        ball_radius_m=0.11,
        config=cfg,
    )
    temporal_errors = []
    x_errors = []
    for fi in fis:
        xyz = result.frames[fi].xyz_world_m
        assert xyz is not None
        temporal_errors.append(float(np.linalg.norm(np.asarray(xyz) - truth[fi])))
        x_errors.append(abs(float(xyz[0]) - float(truth[fi][0])))
    assert float(np.mean(temporal_errors)) < float(np.mean(raw_errors)) * 0.55
    assert float(np.percentile(x_errors, 90)) < 1.0
    assert result.frames[13].center_imputed is True
    assert result.frames[14].center_imputed is True
    assert result.frames[13].status == "VALID_TEMPORAL_3D_INTERPOLATED"
    assert result.diagnostics["frames_with_temporal_3d"] == len(fis)


def test_long_gap_is_not_invented():
    fis = list(range(15))
    camera = _look_at_camera(0)
    cameras = {fi: _look_at_camera(fi) for fi in fis}
    xyz = np.asarray([27.0, 2.0, 0.11])
    center, d = _ideal_diameter(camera, xyz)
    selected = {fi: _candidate(fi, center, d) for fi in fis}
    for fi in range(5, 10):
        selected[fi] = None
    result = refine_temporal_trajectory(
        cameras_by_frame=cameras,
        selected=selected,
        frame_indices=fis,
        fps=30.0,
        config=TemporalRefinementConfig(max_gap_frames=2),
    )
    assert all(result.frames[fi].xyz_world_m is None for fi in range(5, 10))
