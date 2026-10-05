import numpy as np
from stage4_metric3d.backends.sam3d_pitch_refined.pipeline import preflight_v05, run_v05
from stage4_metric3d.backends.sam3d_pitch_refined.config import Sam3DPitchRefinedConfig
from .helpers import build_case


def test_preflight_and_refinement_reduce_sam_translation_error(tmp_path):
    s3,cams,cache,true_trans,frames=build_case(tmp_path/'case')
    pre=preflight_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache)
    assert pre['ready'] is True
    assert pre['camera_convention']['status']=='PASS'
    assert pre['ground_anchor_coverage_at_t0']==1.0
    state=run_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache,output_dir=tmp_path/'out',refine=True)
    tr=state['tracks'][0]
    obs=next(o for o in tr['observations'] if o['frame_index']==11)
    prior=np.asarray(obs['translation']['sam3d_prior_cam_m']); refined=np.asarray(obs['translation']['refined_cam_m'])
    assert np.linalg.norm(refined-true_trans[1]) < np.linalg.norm(prior-true_trans[1])
    assert state['schema_version']=='world-grounded-pose-state-1.1'
    assert state['quality_gates']['implementation_gate']=='PASS'
    assert state['quality_gates']['geometric_quality_gate']=='PASS_SANITY'
    assert (tmp_path/'out'/'stage4_downstream_handoff.json').is_file()


def test_direct_backend_keeps_sam_translation(tmp_path):
    s3,cams,cache,true_trans,frames=build_case(tmp_path/'case')
    state=run_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache,output_dir=tmp_path/'direct',refine=False)
    obs=next(o for o in state['tracks'][0]['observations'] if o['frame_index']==11)
    assert np.allclose(obs['translation']['refined_cam_m'],obs['translation']['sam3d_prior_cam_m'])
    assert state['backend']=='sam3d-direct'


def test_selected_frame_override_targets_exact_zero_based_frame(tmp_path):
    s3,cams,cache,true_trans,frames=build_case(tmp_path/'case')
    pre=preflight_v05(stage3_state=s3,camera_dir=cams,sam3d_cache=cache,selected_frame=12)
    assert pre['ready'] is True
    assert pre['selected_frame']==12
    state=run_v05(
        stage3_state=s3,
        camera_dir=cams,
        sam3d_cache=cache,
        output_dir=tmp_path/'frame12',
        refine=False,
        selected_frame=12,
    )
    assert state['selected_frame']==12
    assert state['provenance']['selected_frame_override']==12
    assert state['tracks'][0]['selected_frame_status']=='VALID'


def test_temporal_status_requires_exact_consecutive_frames(tmp_path):
    s3, cams, cache, true_trans, frames = build_case(tmp_path/'case')
    state = run_v05(
        stage3_state=s3, camera_dir=cams, sam3d_cache=cache,
        output_dir=tmp_path/'nonconsecutive', config=Sam3DPitchRefinedConfig(window_radius_frames=2),
        refine=True,
    )
    track = state['tracks'][0]
    assert track['temporal_status'] == 'APPLIED'
    assert track['temporal_diagnostics']['triplet_count'] == 1
    assert track['temporal_diagnostics']['triplets'] == [[10, 11, 12]]
