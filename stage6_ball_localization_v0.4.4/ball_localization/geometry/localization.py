from __future__ import annotations
from typing import Dict,Optional
import math, numpy as np
from ..camera import CameraStateLite
from ..contracts import BallCandidate2D,BallFrameState

def _angle(a:np.ndarray,b:np.ndarray)->float: return float(math.acos(float(np.clip(np.dot(a,b),-1.0,1.0))))

def angular_radius_from_bbox(camera:CameraStateLite,bbox:list[float])->Optional[Dict[str,float]]:
    x1,y1,x2,y2=[float(x) for x in bbox]
    if x2<=x1 or y2<=y1:return None
    cx,cy=(x1+x2)/2,(y1+y2)/2; pts=np.asarray([[cx,cy],[x1,cy],[x2,cy],[cx,y1],[cx,y2]],float); _,dirs=camera.world_ray(pts)
    ax=0.5*(_angle(dirs[0],dirs[1])+_angle(dirs[0],dirs[2])); ay=0.5*(_angle(dirs[0],dirs[3])+_angle(dirs[0],dirs[4])); alpha=min(ax,ay)
    if not np.isfinite(alpha) or alpha<=1e-8:return None
    return {"alpha_rad":float(alpha),"alpha_x_rad":float(ax),"alpha_y_rad":float(ay),"aspect_ratio":float(max(x2-x1,y2-y1)/max(min(x2-x1,y2-y1),1e-9))}

def _inside(camera:CameraStateLite,xyz:np.ndarray,margin:float)->bool:
    L=float(camera.pitch.get("length_m",105.0)); W=float(camera.pitch.get("width_m",68.0)); return -L/2-margin<=float(xyz[0])<=L/2+margin and -W/2-margin<=float(xyz[1])<=W/2+margin

def _size_prior_validity(*,camera_ok:bool,ang,size_xyz,pitch_ok:bool,height_ok:bool)->str:
    if not camera_ok:return "INVALID_CAMERA"
    if ang is None:return "INVALID_ANGULAR_SIZE"
    if size_xyz is None or not np.all(np.isfinite(size_xyz)):return "INVALID_RAY_OR_DEPTH"
    if not pitch_ok:return "INVALID_PITCH_XY"
    if not height_ok:return "INVALID_HEIGHT"
    return "VALID"

def estimate_frame(camera:CameraStateLite,candidate:BallCandidate2D|None,*,fps:float,ball_radius_m:float=0.11,mode:str="ground-first",ground_size_log_tolerance:float=0.40,pitch_margin_m:float=2.0,max_size_prior_height_m:float=15.0)->BallFrameState:
    ts=camera.timestamp_sec if camera.timestamp_sec is not None else camera.frame_index/float(fps)
    if candidate is None:return BallFrameState(camera.frame_index,float(ts),None,"MISSING",camera.status,"NO_BALL_OBSERVATION",None,None,None,None,None,{"size_prior_validity":"NO_BALL_OBSERVATION"})
    uv=np.asarray(candidate.center_uv,float); ground=camera.intersect_z_plane(uv,ball_radius_m)[0]; contact=ground.copy() if np.all(np.isfinite(ground)) else ground
    if np.all(np.isfinite(contact)):contact[2]=0.0
    ang=angular_radius_from_bbox(camera,candidate.bbox_xyxy); size_xyz=None; size_distance=None
    if ang is not None:
        s=math.sin(ang["alpha_rad"])
        if s>1e-8:
            size_distance=float(ball_radius_m/s); o,d=camera.world_ray(uv); size_xyz=o[0]+size_distance*d[0]
    ground_finite=bool(np.all(np.isfinite(ground)))
    ground_inside=bool(ground_finite and _inside(camera,ground,pitch_margin_m))
    ground_valid=ground_inside
    camera_ok=camera.status in {"VALID","DEGRADED"}
    size_finite=bool(size_xyz is not None and np.all(np.isfinite(size_xyz)))
    size_pitch_ok=bool(size_finite and _inside(camera,size_xyz,pitch_margin_m*3))
    size_height_m=None if not size_finite else float(size_xyz[2])
    size_height_ok=bool(size_finite and ball_radius_m*0.5<=size_height_m<=max_size_prior_height_m)
    size_validity=_size_prior_validity(camera_ok=camera_ok,ang=ang,size_xyz=size_xyz,pitch_ok=size_pitch_ok,height_ok=size_height_ok)
    size_valid=size_validity=="VALID"
    logerr=None
    if ground_valid and ang is not None:
        dist=float(np.linalg.norm(ground-camera.camera_center_world_m))
        if dist>ball_radius_m:
            pred=math.asin(min(0.999999,ball_radius_m/dist)); logerr=abs(math.log(max(ang["alpha_rad"],1e-12)/max(pred,1e-12)))
    consistent=bool(ground_valid and logerr is not None and logerr<=ground_size_log_tolerance); selected=None; method=None
    if mode=="ground-only":
        if ground_valid and camera_ok:selected=ground; method="GROUND_CENTER_Z_EQUALS_RADIUS"; status="VALID_GROUND" if camera.status=="VALID" else "DEGRADED_GROUND_CAMERA"
        else:status="GROUND_LOCALIZATION_INVALID"
    elif mode=="size-prior":
        if size_valid:selected=size_xyz; method="MONOCULAR_BALL_SIZE_PRIOR"; status="VALID_SIZE_PRIOR" if camera.status=="VALID" else "DEGRADED_SIZE_PRIOR_CAMERA"
        else:status="SIZE_PRIOR_INVALID"
    else:
        if consistent and camera_ok:selected=ground; method="GROUND_CENTER_Z_EQUALS_RADIUS"; status="VALID_GROUND" if camera.status=="VALID" else "DEGRADED_GROUND_CAMERA"
        elif ground_valid and camera_ok:status="UNCERTAIN_AIRBORNE_OR_BBOX_SIZE"
        else:status="LOCALIZATION_INVALID"
    diagnostics={
        "ground_hypothesis_valid":ground_valid,
        "ground_intersection_finite":ground_finite,
        "ground_inside_pitch_xy":ground_inside,
        "ground_size_consistent":consistent,
        "ground_size_log_error":logerr,
        "size_prior_valid":size_valid,
        "size_prior_validity":size_validity,
        "size_prior_distance_m":size_distance,
        "size_prior_height_m":size_height_m,
        "size_prior_finite":size_finite,
        "size_prior_inside_pitch_xy":size_pitch_ok,
        "size_prior_height_valid":size_height_ok,
        "angular_size":ang,
        "pitch_prior":candidate.pitch_prior,
        "ranking_score":candidate.ranking_score,
    }
    return BallFrameState(camera.frame_index,float(ts),candidate,"DIRECT",camera.status,status,None if not np.all(np.isfinite(contact)) else contact.astype(float).tolist(),None if not np.all(np.isfinite(ground)) else ground.astype(float).tolist(),None if size_xyz is None or not np.all(np.isfinite(size_xyz)) else size_xyz.astype(float).tolist(),None if selected is None else np.asarray(selected,float).tolist(),method,diagnostics)
