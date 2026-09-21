from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

# Compatibility for the retained v0.2 legacy tests/ablation utilities.
try:
    from benchmark_stage4_legacy3d import synthetic_smoke, build_synthetic_case
except Exception:
    synthetic_smoke = None
    build_synthetic_case = None


def _stats(values):
    a=np.asarray(values,dtype=np.float64); a=a[np.isfinite(a)]
    if not a.size: return {"count":0,"mean":None,"median":None,"p95":None}
    return {"count":int(a.size),"mean":float(a.mean()),"median":float(np.median(a)),"p95":float(np.percentile(a,95))}


def _load_state(path):
    p=Path(path).expanduser().resolve(); d=json.loads(p.read_text(encoding='utf-8'))
    if d.get('schema_version')!='world-grounded-pose-state-1.0':
        raise ValueError(f"Expected world-grounded-pose-state-1.0, got {d.get('schema_version')}")
    return d,p


def evaluate_metric_gt(state, gt_path):
    """Optional exact-schema metric evaluator.

    GT npz contract:
      track_ids: (N,) strings matching Stage4 output order
      frame_indices: (T,) source frame numbers
      joints_world_m: (N,T,25,3)
      roots_world_m: optional (N,T,3)
    """
    tracks=state.get('tracks') or []
    pred_tids=[str(t['track_id']) for t in tracks]
    with np.load(gt_path,allow_pickle=True) as gt:
        tids=[str(x) for x in gt['track_ids'].tolist()]
        frames=[int(x) for x in gt['frame_indices'].tolist()]
        joints=np.asarray(gt['joints_world_m'],dtype=np.float64)
        roots=np.asarray(gt['roots_world_m'],dtype=np.float64) if 'roots_world_m' in gt.files else None
    if tids!=pred_tids:
        raise ValueError(f"GT track order mismatch: expected {pred_tids}, got {tids}")
    if joints.shape!=(len(tids),len(frames),25,3):
        raise ValueError(f"GT joints shape mismatch: {joints.shape}")
    frame_to_t={f:i for i,f in enumerate(frames)}
    mpjpe=[]; xerr=[]; rooterr=[]; rootx=[]; per_joint=[[] for _ in range(25)]
    for n,tr in enumerate(tracks):
        for obs in tr.get('observations') or []:
            f=int(obs['frame_index']); ti=frame_to_t.get(f)
            if ti is None or not obs.get('valid'): continue
            pred=np.asarray([j.get('xyz_world_m') if j.get('xyz_world_m') is not None else [np.nan]*3 for j in obs.get('joints',[])],dtype=float)
            ref=joints[n,ti]
            mask=np.isfinite(pred).all(axis=1)&np.isfinite(ref).all(axis=1)
            if np.any(mask):
                e=np.linalg.norm(pred[mask]-ref[mask],axis=1); mpjpe.extend(e.tolist())
                dx=np.abs(pred[mask,0]-ref[mask,0]); xerr.extend(dx.tolist())
                for j in np.flatnonzero(mask): per_joint[j].append(float(abs(pred[j,0]-ref[j,0])))
            if roots is not None:
                pr=np.asarray((obs.get('root') or {}).get('pred_world_m') or [np.nan]*3,dtype=float); rr=roots[n,ti]
                if np.isfinite(pr).all() and np.isfinite(rr).all():
                    rooterr.append(float(np.linalg.norm(pr-rr))); rootx.append(float(abs(pr[0]-rr[0])))
    return {
        "WorldMPJPE_m":_stats(mpjpe),
        "PerJointLongitudinalAbsError_m": {str(i):_stats(v) for i,v in enumerate(per_joint)},
        "AllJointLongitudinalAbsError_m":_stats(xerr),
        "RootError_m":_stats(rooterr),
        "RootLongitudinalAbsError_m":_stats(rootx),
    }


def main():
    p=argparse.ArgumentParser(description='Evaluate Stage4 v0.4 world-grounded pose state')
    p.add_argument('--state',required=True)
    p.add_argument('--metric-gt-npz',default=None,help='Optional exact 25-joint metric GT contract')
    p.add_argument('--output',default=None)
    args=p.parse_args()
    state,path=_load_state(args.state)
    report={
        "schema_version":"stage4-v040-evaluation-report-1.0",
        "state":str(path),
        "stage4_version":state.get('stage4_version'),
        "method":state.get('method'),
        "quality_gates":state.get('quality_gates'),
        "metrics_without_gt":state.get('metrics'),
        "camera_domain":state.get('camera_domain'),
        "metric_accuracy":"NOT_EVALUATED" if not args.metric_gt_npz else "EVALUATED",
    }
    if args.metric_gt_npz:
        report['metric_gt']=evaluate_metric_gt(state,args.metric_gt_npz)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
