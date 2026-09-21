from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage_1_camera.contracts import CameraState, CameraStatus
from stage_1_camera.visualization import save_camera_3d_view
from stage_1_camera.evaluation.offside_geometry import ground_longitudinal_error, vertical_plane_projection_error


def look_at(C,target=np.zeros(3),up=np.array([0.,0.,1.])):
    f=(target-C);f/=np.linalg.norm(f);r=np.cross(f,up);r/=np.linalg.norm(r);d=np.cross(f,r);d/=np.linalg.norm(d)
    return np.stack([r,d,f])

C=np.array([0.,-65.,22.]); K=np.array([[1400.,0.,960.],[0.,1400.,540.],[0.,0.,1.]])
cam=CameraState(100,1920,1080,K,look_at(C),C,status=CameraStatus.VALID)
out=Path(__file__).resolve().parent/'synthetic_camera_3d.png'; save_camera_3d_view(cam,str(out))
pts=np.array([[-40,0,0],[-20,10,0],[0,0,0],[20,-10,0],[40,0,0]],float)
print('Ground round-trip / GLE-X self-check:',ground_longitudinal_error(cam,cam,pts))
print('VPPE self-check:',vertical_plane_projection_error(cam,cam,[-30,0,30]))
print('Saved:',out)
