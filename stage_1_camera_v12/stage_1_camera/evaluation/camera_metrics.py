from __future__ import annotations
import numpy as np
from ..contracts import CameraState


def rotation_error_deg(R_pred, R_gt) -> float:
    R=np.asarray(R_pred)@np.asarray(R_gt).T
    v=np.clip((np.trace(R)-1)/2,-1,1)
    return float(np.degrees(np.arccos(v)))


def camera_center_error_m(C_pred, C_gt) -> float:
    return float(np.linalg.norm(np.asarray(C_pred)-np.asarray(C_gt)))


def reprojection_error_px(cam: CameraState, xyz, uv_gt, reducer='median') -> float:
    pred=cam.project_world(np.asarray(xyz,float)); gt=np.asarray(uv_gt,float)
    e=np.linalg.norm(pred-gt,axis=1)
    return float(np.mean(e) if reducer=='mean' else np.median(e))
