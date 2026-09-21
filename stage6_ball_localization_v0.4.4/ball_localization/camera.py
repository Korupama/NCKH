from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple
import json, cv2, numpy as np

from .coordinates import soccernet_rotation_to_stage6, soccernet_xyz_to_stage6

@dataclass
class CameraStateLite:
    frame_index:int; image_width:int; image_height:int; K:np.ndarray; R_world_to_camera:np.ndarray; camera_center_world_m:np.ndarray; distortion:np.ndarray; status:str; pitch:Dict[str,Any]; timestamp_sec:float|None=None
    @property
    def t_world_to_camera(self)->np.ndarray: return -self.R_world_to_camera @ self.camera_center_world_m
    def undistort_pixels(self,uv:np.ndarray)->np.ndarray:
        pts=np.asarray(uv,dtype=np.float64); pts=pts[None,:] if pts.ndim==1 else pts
        return cv2.undistortPoints(pts.reshape(-1,1,2),self.K,self.distortion,P=self.K).reshape(-1,2)
    def world_ray(self,uv:np.ndarray)->Tuple[np.ndarray,np.ndarray]:
        pts=self.undistort_pixels(uv); invK=np.linalg.inv(self.K)
        dirs_cam=(invK @ np.column_stack([pts,np.ones(len(pts))]).T).T
        dirs=(self.R_world_to_camera.T @ dirs_cam.T).T; dirs/=np.linalg.norm(dirs,axis=1,keepdims=True).clip(min=1e-12)
        return np.repeat(self.camera_center_world_m[None,:],len(pts),axis=0),dirs
    def intersect_z_plane(self,uv:np.ndarray,z_m:float)->np.ndarray:
        o,d=self.world_ray(uv); den=d[:,2]; lam=np.full(len(d),np.nan); good=np.abs(den)>1e-10; lam[good]=(float(z_m)-o[good,2])/den[good]; good &= lam>0
        xyz=o+lam[:,None]*d; xyz[~good]=np.nan; return xyz
    def intersect_y_plane(self, uv: np.ndarray, y_m: float) -> np.ndarray:
        """Forward-ray intersection; parallel/behind/nonfinite rays return NaN."""
        o, d = self.world_ray(uv)
        lam = np.full(len(d), np.nan)
        good = np.abs(d[:, 1]) > 1e-10
        lam[good] = (float(y_m) - o[good, 1]) / d[good, 1]
        xyz = o + lam[:, None] * d
        xyz[(lam <= 0) | ~np.isfinite(lam)] = np.nan
        return xyz
    def project_world(self,xyz:np.ndarray,distort:bool=True)->np.ndarray:
        pts=np.asarray(xyz,dtype=np.float64); pts=pts[None,:] if pts.ndim==1 else pts
        rvec,_=cv2.Rodrigues(self.R_world_to_camera); tvec=self.t_world_to_camera.reshape(3,1); dist=self.distortion if distort else np.zeros_like(self.distortion)
        uv,_=cv2.projectPoints(pts,rvec,tvec,self.K,dist); return uv.reshape(-1,2)

def _dist(data:Dict[str,Any])->np.ndarray:
    radial=list(data.get("radial") or [])+[0.0]*6; tang=list(data.get("tangential") or [])+[0.0]*2; thin=list(data.get("thin_prism") or [])+[0.0]*4
    return np.asarray([radial[0],radial[1],tang[0],tang[1],radial[2],radial[3],radial[4],radial[5],*thin[:4]],dtype=np.float64)

def load_camera_state(path:str|Path)->CameraStateLite:
    d=json.loads(Path(path).read_text(encoding="utf-8")); image=d.get("image") or {}
    if str(d.get("schema_version"))!="1.2": raise ValueError("Stage-1 CameraState schema 1.2 required")
    if image.get("pixel_space")!="original_raw": raise ValueError("Stage-1 original_raw pixel space required")
    return CameraStateLite(int(d["frame_index"]),int(image["width"]),int(image["height"]),np.asarray(d["intrinsics"]["K"],float).reshape(3,3),np.asarray(d["extrinsics"]["R_world_to_camera"],float).reshape(3,3),np.asarray(d["extrinsics"]["camera_center_world_m"],float).reshape(3),_dist(d.get("distortion") or {}),str(d.get("status","INVALID")),dict(d.get("pitch") or {}),None if d.get("timestamp_sec") is None else float(d["timestamp_sec"]))

def optimized_camera_dir(stage1_root:str|Path)->Path:
    root=Path(stage1_root).expanduser().resolve(); candidates=[root/"outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states",root/"stage_1_camera/outputs/temporal_v13/batch_video/shot_ptz/optimized_camera_states"]
    for c in candidates:
        if c.is_dir(): return c
    raise FileNotFoundError("optimized_camera_states not found")

def camera_state_for_frame(stage1_root:str|Path,frame_index:int)->CameraStateLite:
    p=optimized_camera_dir(stage1_root)/f"camera_state_{int(frame_index):08d}.json"
    if not p.is_file(): raise FileNotFoundError(p)
    return load_camera_state(p)

def camera_from_soccernet_calibration(calib: Dict[str, Any], *, frame_index: int = 0, image_width: int | None = None, image_height: int | None = None) -> CameraStateLite:
    """Build canonical Stage-6 camera from a SoccerNet calibration dictionary.

    SoccerNet pitch geometry uses negative Z for elevated goal points. Stage 6
    freezes +Z as upward. We therefore canonicalize the *world frame* with the
    proper rotation T=diag(1,-1,-1), preserving X exactly while flipping Y/Z.
    Projection is invariant because camera center and world->camera rotation are
    transformed together.
    """
    pp = calib.get("principal_point") or [None, None]
    cx = float(pp[0]); cy = float(pp[1])
    width = int(image_width if image_width is not None else round(2 * cx))
    height = int(image_height if image_height is not None else round(2 * cy))
    fx = float(calib["x_focal_length"]); fy = float(calib["y_focal_length"])
    K = np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    if calib.get("rotation_matrix") is not None:
        R_sn = np.asarray(calib["rotation_matrix"], dtype=np.float64).reshape(3,3)
    else:
        pan = float(calib["pan_degrees"]) * np.pi / 180.0
        tilt = float(calib["tilt_degrees"]) * np.pi / 180.0
        roll = float(calib["roll_degrees"]) * np.pi / 180.0
        Rpan=np.asarray([[np.cos(pan),-np.sin(pan),0],[np.sin(pan),np.cos(pan),0],[0,0,1]],float)
        Rtilt=np.asarray([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]],float)
        Rroll=np.asarray([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]],float)
        R_sn = (Rpan @ Rtilt @ Rroll).T
    radial=list(calib.get("radial_distortion") or [])+[0.0]*6
    tang=list(calib.get("tangential_distortion") or [])+[0.0]*2
    thin=list(calib.get("thin_prism_distortion") or [])+[0.0]*4
    distortion=np.asarray([radial[0],radial[1],tang[0],tang[1],radial[2],radial[3],radial[4],radial[5],*thin[:4]],dtype=np.float64)
    R = soccernet_rotation_to_stage6(R_sn)
    C = soccernet_xyz_to_stage6(calib["position_meters"])
    # Guard against accidentally constructing an improper matrix; cv2.Rodrigues
    # assumes a proper SO(3) rotation.
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-6):
        raise ValueError(f"Canonicalized SoccerNet camera rotation is not proper: det={np.linalg.det(R)}")
    return CameraStateLite(frame_index,width,height,K,R,C,distortion,"VALID",{"length_m":105.0,"width_m":68.0},None)
