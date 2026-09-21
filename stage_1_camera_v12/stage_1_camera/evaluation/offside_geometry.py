from __future__ import annotations
from typing import Dict, Iterable, Sequence
import numpy as np
from ..contracts import CameraState


def ground_longitudinal_error(pred: CameraState, gt: CameraState,
                              xyz_ground: np.ndarray) -> Dict[str,float]:
    """GLE-X: render ground points with GT camera, unproject with predicted camera, compare X."""
    pts=np.asarray(xyz_ground,float)
    if pts.ndim==1: pts=pts[None,:]
    if np.max(np.abs(pts[:,2]))>1e-8: raise ValueError('xyz_ground must have Z=0')
    uv=gt.project_world(pts)
    rec=pred.intersect_pitch(uv)
    err=np.abs(rec[:,0]-pts[:,0])
    err=err[np.isfinite(err)]
    return {"count":int(len(err)),"mean_m":float(np.mean(err)) if len(err) else float('nan'),
            "median_m":float(np.median(err)) if len(err) else float('nan'),
            "p95_m":float(np.percentile(err,95)) if len(err) else float('nan')}


def _polyline_distance(a,b):
    # Same parameterization of Y/Z samples => pointwise image discrepancy.
    good=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    if not np.any(good): return np.array([])
    return np.linalg.norm(a[good]-b[good],axis=1)


def vertical_plane_projection_error(pred: CameraState, gt: CameraState, x_values: Sequence[float],
                                    z_values=(0.0,0.5,1.0,1.5,2.0,2.5), y_samples: int=31) -> Dict[str,float]:
    """VPPE: pixel discrepancy of vertical X=constant plane samples under predicted vs GT camera."""
    errors=[]
    ys=np.linspace(-gt.pitch.width_m/2,gt.pitch.width_m/2,y_samples)
    for x in x_values:
        pts=np.array([[x,y,z] for z in z_values for y in ys],float)
        e=_polyline_distance(pred.project_world(pts),gt.project_world(pts))
        errors.extend(e.tolist())
    e=np.asarray(errors,float)
    return {"count":int(len(e)),"mean_px":float(np.mean(e)) if len(e) else float('nan'),
            "median_px":float(np.median(e)) if len(e) else float('nan'),
            "p95_px":float(np.percentile(e,95)) if len(e) else float('nan')}
