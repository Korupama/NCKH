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


def filter_pitch_candidates(candidates, camera, *, margin_m=0.0):
    """Keep centres whose forward ground-ray hit is within the calibrated pitch.

    This is an image/ground-footprint constraint, not an airborne 3D test.
    Unverifiable candidates are rejected in this explicitly strict mode.
    """
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError('candidate pitch margin must be finite and nonnegative')
    kept, rejected = [], []
    for candidate in candidates:
        xyz = None
        reason = None
        if camera is None or str(camera.status).upper() not in {'VALID', 'DEGRADED'}:
            reason = 'PITCH_CAMERA_UNAVAILABLE'
        else:
            length = float(camera.pitch.get('length_m', 105.0))
            width = float(camera.pitch.get('width_m', 68.0))
            if not np.isfinite([length, width]).all() or min(length, width) <= 0:
                reason = 'INVALID_PITCH_DIMENSIONS'
            else:
                try:
                    hit = camera.intersect_z_plane(np.asarray(candidate.center_uv, float), 0.0)[0]
                    if not np.isfinite(hit).all():
                        reason = 'NO_FORWARD_PITCH_INTERSECTION'
                    else:
                        xyz = hit.tolist()
                        # Include the boundary, with only numerical roundoff tolerance.
                        if abs(hit[0]) > length / 2 + margin_m + 1e-8 or abs(hit[1]) > width / 2 + margin_m + 1e-8:
                            reason = 'OUTSIDE_PITCH'
                except (ValueError, np.linalg.LinAlgError):
                    reason = 'INVALID_PITCH_PROJECTION'
        evidence = {'ground_intersection_xyz_m': xyz, 'margin_m': float(margin_m),
                    'camera_status': None if camera is None else camera.status}
        if reason:
            rejected.append({'candidate': candidate.to_dict(), 'reason': reason, **evidence})
        else:
            candidate.metadata['pitch_candidate_filter'] = {'policy': 'strict-ground', **evidence}
            kept.append(candidate)
    return kept, rejected
