from __future__ import annotations
from typing import Iterable
import numpy as np
from .camera import CameraStateLite
from .contracts import BallCandidate2D

def pitch_soft_prior(camera:CameraStateLite,center_uv:Iterable[float],*,margin_m:float=12.0,far_prior:float=0.20)->float:
    xyz=camera.intersect_z_plane(np.asarray(center_uv,float),0.0)[0]
    if not np.all(np.isfinite(xyz)): return float(far_prior)
    L=float(camera.pitch.get("length_m",105.0)); W=float(camera.pitch.get("width_m",68.0)); dx=max(0.0,abs(float(xyz[0]))-L/2); dy=max(0.0,abs(float(xyz[1]))-W/2); d=float(np.hypot(dx,dy))
    if d<=0: return 1.0
    if d>=margin_m: return float(far_prior)
    return float(1.0-(1.0-far_prior)*(d/margin_m))

def apply_pitch_prior(candidates:list[BallCandidate2D],camera:CameraStateLite,*,margin_m:float=12.0,far_prior:float=0.20)->list[BallCandidate2D]:
    for c in candidates:
        c.pitch_prior=pitch_soft_prior(camera,c.center_uv,margin_m=margin_m,far_prior=far_prior); c.ranking_score=float(c.detector_score)*c.pitch_prior; c.metadata["pitch_soft_prior"]=c.pitch_prior
    return candidates
