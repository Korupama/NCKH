import numpy as np
import pytest
from test_contact_v050 import fixture, camera
from ball_localization.contact import apply_contact

@pytest.mark.parametrize('quality',['VALID','DEGRADED'])
def test_foot_without_anchor(quality):
    s,p,_=fixture(); c=camera(); c.status=quality
    result=apply_contact(s,p,None,c); r=result['selected_frame_ball']
    assert r['contact']['status']=='SUPPORTED'
    assert r['status']==quality+'_CONTACT_GROUND_PLANE'
    assert r['method']=='CONTACT_GROUND_PLANE' and not r['fallback']['used']
    assert r['fallback']['reason'] is None
    assert not r['localization']['ground_anchor_required'] and not r['localization']['ground_anchor_used']
    np.testing.assert_allclose(r['center_xyz_world_m'],[0,0,.11],atol=1e-12)
    assert result['status']==quality
    assert result['diagnostics']['v051_gates']['research_accuracy_frozen'] is False

def test_foot_ignores_irrelevant_anchor():
    s,p,h=fixture(); h['selected_frame']=999
    assert not apply_contact(s,p,h,camera())['selected_frame_ball']['fallback']['used']

def test_degraded_vertical_rejected():
    s,p,h=fixture('HEAD'); c=camera(); c.status='DEGRADED'
    r=apply_contact(s,p,h,c)['selected_frame_ball']
    assert r['contact']['track_id']=='p1'
    assert r['fallback']['used'] and r['fallback']['reason']=='VERTICAL_CAMERA_REQUIRES_VALID'

def test_missing_vertical_anchor():
    s,p,h=fixture('HEAD')
    r=apply_contact(s,p,None,camera())['selected_frame_ball']
    assert r['fallback']['reason']=='VERTICAL_GROUND_ANCHOR_UNUSABLE'

@pytest.mark.parametrize('xyz',[[np.nan,0,.11],[60,0,.11],[0,40,.11]])
def test_bad_ground_intersection(xyz):
    s,p,h=fixture(); c=camera()
    c.intersect_z_plane=lambda *args:np.array([xyz])
    r=apply_contact(s,p,None,c)['selected_frame_ball']
    assert r['fallback']['used'] and r['fallback']['reason']=='GROUND_INTERSECTION_INVALID_OR_OUTSIDE_PITCH'

def test_invalid_camera_rejected():
    s,p,h=fixture(); c=camera(); c.status='INVALID'
    r=apply_contact(s,p,None,c)['selected_frame_ball']
    assert r['fallback']['reason']=='GROUND_CAMERA_UNUSABLE'

def test_ground_fields_recomputed_together():
    s,p,h=fixture(); s['selected_frame_ball']['ground_center_xyz_world_m']=[999,999,.11]
    result=apply_contact(s,p,None,camera()); r=result['selected_frame_ball']
    assert r['ground_center_xyz_world_m']==r['center_xyz_world_m']
    assert result['frames'][1]['ground_center_xyz_world_m']==r['center_xyz_world_m']
