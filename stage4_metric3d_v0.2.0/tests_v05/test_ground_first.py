import numpy as np
from .helpers import build_case
from stage4_metric3d.backends.sam3d_pitch_refined.pipeline import run_v05, preflight_v05
from stage4_metric3d.backends.sam3d_pitch_refined.config import Sam3DPitchRefinedConfig
from stage4_metric3d.backends.sam3d_pitch_refined.quality import ground_quality_reasons


def test_large_sam_error_does_not_veto_ground(tmp_path):
    s3, cams, cache, truth, _ = build_case(tmp_path/'case', prior_offset_cam=(4., -3., 5.))
    cfg = Sam3DPitchRefinedConfig(window_radius_frames=0)
    pre = preflight_v05(stage3_state=s3, camera_dir=cams, sam3d_cache=cache, config=cfg)
    assert pre['ground_anchor_coverage_at_t0'] == 1.
    assert pre['temporal_status'] == 'TEMPORAL_NOT_AVAILABLE'
    state = run_v05(stage3_state=s3, camera_dir=cams, sam3d_cache=cache, output_dir=tmp_path/'out', config=cfg)
    obs = state['tracks'][0]['observations'][0]
    assert obs['quality']['initialization'] == 'GROUND_CONSENSUS'
    assert obs['quality']['ground_vs_sam_disagreement'] == 'LARGE_SAM_DEPTH_ERROR'
    assert obs['quality']['ground_vs_refined_m'] <= .75 + 1e-8
    assert np.linalg.norm(np.array(obs['translation']['refined_cam_m'])-truth[1]) < .05
    assert state['quality_gates']['geometric_quality_gate'] == 'PASS_SANITY'


def test_old_frame104_ground_metrics_fail():
    reasons = ground_quality_reasons({'ground_contact_residual_cm': {'median':46.35,'p95':135.05}}, .20)
    assert len(reasons) == 3


def test_missing_ground_metrics_cannot_pass():
    assert len(ground_quality_reasons({}, None)) == 3
