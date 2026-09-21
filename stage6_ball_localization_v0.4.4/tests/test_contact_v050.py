import copy
import numpy as np
import pytest
from ball_localization.camera import CameraStateLite
from ball_localization.contact import apply_contact, associate_contact, ground_anchor, plane_sensitivity, validate_replay
from ball_localization.contracts import BallCandidate2D
from ball_localization.providers.fusion import fuse_candidates

def camera():
    return CameraStateLite(1,100,100,np.eye(3),np.eye(3),np.array([0.,-5.,-1.]),np.zeros(5),'VALID',{})

def fixture(region='FOOT'):
    replay=dict(image_width=100,image_height=100,selected_frame=1,fps=30,coordinate_space='RAW_DISTORTED_PIXEL',video_path='test.mp4')
    frames=[]; observations=[]
    for fi,offset in [(0,5),(1,0),(2,5)]:
        uv=[0.,5/1.11] if region=='FOOT' else [0.,2.5]
        cand=dict(frame_index=fi,candidate_id=str(fi),center_uv=uv,diameter_px=2,source='test')
        frames.append(dict(frame_index=fi,candidate=cand,selected_center_xyz_world_m=[0.,0.,.11],diagnostics={}))
        names=['left_ankle','left_heel','left_big_toe','left_small_toe'] if region=='FOOT' else ['nose','left_eye']
        observations.append(dict(frame_index=fi,keypoints_133=[dict(name=n,x=uv[0]+offset,y=uv[1],state='VALID') for n in names]))
    source=dict(replay_context=replay,frames=frames,ball_radius_m=.11,selected_frame_ball=dict(frame_index=1,center_xyz_world_m=[0.,0.,.11],method='HYBRID_GROUND_PLANE'))
    poses=dict(replay_context=copy.deepcopy(replay),tracks=[dict(track_id='p1',observations=observations)])
    hand=dict(selected_frame=1,coordinate_frame=dict(units='metres',origin='center',x_axis='goal_to_goal',y_axis='touchline_to_touchline',z_axis='up'),tracks=[dict(track_id='p1',selected_frame_ground_anchor=dict(status='VALID',method='DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN',xyz_ground_m=[0.,0.,0.]))])
    return source,poses,hand

def test_y_plane():
    c=camera(); assert np.allclose(c.intersect_y_plane([0,2.5],0)[0],[0,0,1])
    assert np.isnan(c.intersect_y_plane([0,0],0)).all()
    assert np.isnan(c.intersect_y_plane([0,-1],0)).all()

@pytest.mark.parametrize('region,method',[('FOOT','CONTACT_GROUND_PLANE'),('HEAD','CONTACT_VERTICAL_PLANE')])
def test_contact(region,method):
    s,p,h=fixture(region); original=copy.deepcopy(s)
    r=apply_contact(s,p,h,camera()); sel=r['selected_frame_ball']
    assert sel['method']==method and not sel['fallback']['used']
    assert sel['contact']['track_id']=='p1'
    assert r['frames'][0]==s['frames'][0] and r['frames'][2]==s['frames'][2]
    assert s==original
    assert sel['X_world_m']==r['frames'][1]['selected_center_xyz_world_m'][0]

def test_ambiguous():
    s,p,h=fixture(); p['tracks'].append(dict(p['tracks'][0],track_id='p2'))
    r=apply_contact(s,p,h,camera())['selected_frame_ball']
    assert r['contact']['status']=='AMBIGUOUS' and r['contact']['track_id'] is None and r['fallback']['used']

def test_no_temporal_support():
    s,p,h=fixture(); p['tracks'][0]['observations']=p['tracks'][0]['observations'][1:2]
    assert apply_contact(s,p,h,camera())['selected_frame_ball']['contact']['status']=='INSUFFICIENT_TEMPORAL_SUPPORT'

@pytest.mark.parametrize('status,method',[('DEGRADED','DENSEST_ANKLE_FOOT_CLUSTER_MEDIAN'),('VALID','HEAD_HEIGHT_PROXY')])
def test_unsafe_anchor_fallback(status,method):
    s,p,h=fixture('HEAD'); h['tracks'][0]['selected_frame_ground_anchor'].update(status=status,method=method)
    assert apply_contact(s,p,h,camera())['selected_frame_ball']['fallback']['used']

def test_replay_mismatch():
    s,p,h=fixture(); p['replay_context']['fps']=25
    with pytest.raises(ValueError): apply_contact(s,p,h,camera())

def test_sensitivity():
    d=plane_sensitivity(camera(),[1,2.5],0)
    assert d['max_abs_delta_x_m']==pytest.approx(.2)
    assert len(d['samples'])==4

def test_fusion():
    a=BallCandidate2D(1,'a',[0,0,10,10],[5,5],.5,'yolo',10)
    b=BallCandidate2D(1,'b',[1,1,10,10],[5.5,5.5],.9,'stage2',9)
    c=BallCandidate2D(1,'c',[50,50,60,60],[55,55],.5,'stage2',10)
    fused,counts=fuse_candidates({1:[a]},{1:[b,c]})
    assert len(fused[1])==2 and counts['duplicates']==1
    assert fused[1][0].diameter_px==10 and len(fused[1][0].metadata['sources'])==2
    assert not a.metadata

def test_missing_ball():
    s,p,h=fixture(); s['frames'][1]['candidate']=None
    assert apply_contact(s,p,h,camera())['selected_frame_ball']['contact']['status']=='NO_BALL_OBSERVATION'

def test_invalid_baseline_not_usable():
    s,p,h=fixture(); p['tracks']=[]; s['selected_frame_ball']['center_xyz_world_m']=[float('nan'),0,0]
    r=apply_contact(s,p,h,camera())['selected_frame_ball']
    assert r['center_xyz_world_m'] is None and not r['localization']['usable_for_offside']

def test_metrics_do_not_invent_gt():
    from ball_localization.evaluation.contact_metrics import evaluate_contact_rows
    assert evaluate_contact_rows([])['contact_track_accuracy'] is None
    r=evaluate_contact_rows([dict(gt_x=1,predicted_x=1.2,gt_track='a',predicted_track=None),dict(gt_x=2,predicted_x=None)])
    assert r['geometry_coverage']==.5 and r['contact_track_accuracy']==0
    assert r['BLE_X_MAE_m']==pytest.approx(.2)
