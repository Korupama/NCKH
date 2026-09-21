import numpy as np

from stage_1_camera.contracts import CameraState, CameraStatus, CameraTimeline
from stage_1_camera.quality_gate import CameraQualityGate
from stage_1_camera.temporal import rescue_degraded_bracketed


def look_at(C, target=np.array([0.,0.,0.]), up=np.array([0.,0.,1.])):
    f=(target-C); f=f/np.linalg.norm(f)
    r=np.cross(f,up); r=r/np.linalg.norm(r)
    d=np.cross(f,r); d=d/np.linalg.norm(d)
    return np.stack([r,d,f])


def make(frame, C, mode='full', ransac=0):
    K=np.array([[1600.,0,960.],[0,1600.,540.],[0,0,1.]])
    cam=CameraState(frame,1920,1080,K,look_at(np.array(C,dtype=float)),np.array(C,dtype=float),
                    status=CameraStatus.DEGRADED,source={'backend':'pnlcalib','pnl_refine':False},
                    evidence={'rep_err_px':2.0,'mode':mode,'use_ransac':ransac,'num_keypoints_used':10,'num_lines_used':0,
                              'keypoint_image_convex_hull_area_ratio':0.08,'keypoint_image_x_span_ratio':0.5,
                              'keypoint_image_y_span_ratio':0.2,'keypoint_image_quadrants_occupied':3},
                    diagnostics={'solver_status':'SOLVED'})
    CameraQualityGate().evaluate(cam)
    return cam


def test_bracketed_temporal_rescue_promotes_when_target_reprojection_validates():
    left=make(0,[-1.,-60.,14.])
    right=make(2,[1.,-60.,14.])
    target=make(1,[0.,-60.,14.],mode='ground_plane',ransac=5)
    # Target correspondences come from the true midpoint camera, not the degraded direct camera.
    true_mid=CameraState(1,1920,1080,target.K,look_at(np.array([0.,-60.,14.])),np.array([0.,-60.,14.]),status=CameraStatus.VALID)
    world=np.array([[-30,-20,0],[-15,0,0],[0,15,0],[20,-10,0],[35,10,0],[-10,5,2.44],[10,-5,2.44]],dtype=float)
    uv=true_mid.project_world(world)
    target.evidence['keypoint_correspondences']=[
        {'id':i+1,'world_xyz_m':X.tolist(),'uv_px':u.tolist()} for i,(X,u) in enumerate(zip(world,uv))
    ]
    out=rescue_degraded_bracketed(CameraTimeline([left,target,right],shot_id='shotA'),target_frame=1)
    rescued=out.by_frame(1)
    assert rescued.temporal['rescue']['validated'] is True
    assert rescued.status == CameraStatus.VALID
    assert rescued.diagnostics['capabilities']['vertical_3d']['status'] == 'VALID'
    assert rescued.diagnostics['capabilities']['offside_3d_ready'] is True
