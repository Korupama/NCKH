from pathlib import Path
import os, pytest
import math
import numpy as np
from ball_localization.camera import load_camera_state
from ball_localization.contracts import BallCandidate2D
from ball_localization.geometry import estimate_frame

STAGE1_ROOT=Path(os.environ.get('STAGE1_ROOT','/mnt/data/_stage1/stage_1_camera_v12'))
CAM=STAGE1_ROOT/'outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states/camera_state_00000086.json'
if not CAM.is_file():
    pytest.skip('Set STAGE1_ROOT to run real Stage-1 camera geometry tests', allow_module_level=True)

def point_on_principal_ray(camera,z):
    d=camera.R_world_to_camera.T @ np.array([0.0,0.0,1.0]); lam=(z-camera.camera_center_world_m[2])/d[2]; return camera.camera_center_world_m+lam*d

def ideal_bbox(camera,xyz,radius=0.11):
    L=float(np.linalg.norm(xyz-camera.camera_center_world_m)); alpha=math.asin(radius/L); r=float(camera.K[0,0])*math.tan(alpha); c=camera.project_world(xyz)[0]; return [float(c[0]-r),float(c[1]-r),float(c[0]+r),float(c[1]+r)],c.tolist()

def test_ground_center_recovers_xyz():
    c=load_camera_state(CAM); truth=point_on_principal_ray(c,0.11); box,uv=ideal_bbox(c,truth); cand=BallCandidate2D(86,'x',box,uv,.9,'synthetic',box[2]-box[0]); state=estimate_frame(c,cand,fps=30,mode='ground-only'); assert np.linalg.norm(np.asarray(state.selected_center_xyz_world_m)-truth)<1e-5

def test_size_prior_recovers_airborne_on_axis():
    c=load_camera_state(CAM); truth=point_on_principal_ray(c,3.0); box,uv=ideal_bbox(c,truth); cand=BallCandidate2D(86,'x',box,uv,.9,'synthetic',box[2]-box[0]); state=estimate_frame(c,cand,fps=30,mode='size-prior'); assert np.linalg.norm(np.asarray(state.selected_center_xyz_world_m)-truth)<0.05
