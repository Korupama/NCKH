from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np


def _paired(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape or p.shape[-1] != 3:
        raise ValueError(f"Prediction/GT shape mismatch: {p.shape} vs {g.shape}")
    mask = np.isfinite(p).all(axis=-1) & np.isfinite(g).all(axis=-1)
    return p, g, mask


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g, mask = _paired(pred, gt)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.linalg.norm(p[mask] - g[mask], axis=-1)))


def root_aligned_mpjpe(pred: np.ndarray, gt: np.ndarray, root_indices: Sequence[int] = (11, 12)) -> float:
    p = np.asarray(pred, dtype=np.float64).copy()
    g = np.asarray(gt, dtype=np.float64).copy()
    if p.ndim == 2:
        p, g = p[None], g[None]
    errors = []
    for pi, gi in zip(p, g):
        roots = [j for j in root_indices if np.isfinite(pi[j]).all() and np.isfinite(gi[j]).all()]
        if not roots:
            continue
        pr = np.mean(pi[roots], axis=0)
        gr = np.mean(gi[roots], axis=0)
        mask = np.isfinite(pi).all(axis=1) & np.isfinite(gi).all(axis=1)
        if np.any(mask):
            errors.extend(np.linalg.norm((pi[mask] - pr) - (gi[mask] - gr), axis=1).tolist())
    return float(np.mean(errors)) if errors else float("nan")


def procrustes_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim == 2:
        p, g = p[None], g[None]
    errs = []
    for pi, gi in zip(p, g):
        mask = np.isfinite(pi).all(axis=1) & np.isfinite(gi).all(axis=1)
        X, Y = pi[mask], gi[mask]
        if len(X) < 3:
            continue
        muX, muY = X.mean(0), Y.mean(0)
        X0, Y0 = X - muX, Y - muY
        normX, normY = np.linalg.norm(X0), np.linalg.norm(Y0)
        if normX < 1e-12 or normY < 1e-12:
            continue
        Xn, Yn = X0 / normX, Y0 / normY
        U, s, Vt = np.linalg.svd(Xn.T @ Yn)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = U @ Vt
        scale = float(np.sum(s) * normY / normX)
        aligned = scale * X0 @ R + muY
        errs.extend(np.linalg.norm(aligned - Y, axis=1).tolist())
    return float(np.mean(errs)) if errs else float("nan")


def longitudinal_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p, g, mask = _paired(pred, gt)
    out = np.full(mask.shape, np.nan, dtype=np.float64)
    out[mask] = np.abs(p[..., 0][mask] - g[..., 0][mask])
    return out


def group_longitudinal_mae(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    e = longitudinal_errors(pred, gt)
    groups = {
        "head": (0, 1, 2, 3, 4),
        "shoulder": (5, 6),
        "hip": (11, 12),
        "knee": (13, 14),
        "ankle": (15, 16),
        "foot": (17, 18, 19, 20, 21, 22),
    }
    result = {}
    for name, idx in groups.items():
        vals = e[..., list(idx)].reshape(-1)
        vals = vals[np.isfinite(vals)]
        result[name] = float(np.mean(vals)) if vals.size else float("nan")
    return result


def goalward_anchor_longitudinal_error(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    attack_sign: int,
    legal_mask: Optional[Sequence[bool]] = None,
) -> float:
    if attack_sign not in (-1, 1):
        raise ValueError("attack_sign must be -1 or +1")
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.ndim != 2 or p.shape != g.shape:
        raise ValueError("GALE expects one Jx3 pose")
    mask = np.isfinite(p).all(axis=1) & np.isfinite(g).all(axis=1)
    if legal_mask is not None:
        mask &= np.asarray(legal_mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    qp = attack_sign * p[mask, 0]
    qg = attack_sign * g[mask, 0]
    return float(abs(float(np.max(qp)) - float(np.max(qg))))


def pairwise_longitudinal_ordering_accuracy(
    pred_x: Sequence[float],
    gt_x: Sequence[float],
    *,
    max_gt_separation_m: Optional[float] = None,
) -> float:
    p = np.asarray(pred_x, dtype=np.float64)
    g = np.asarray(gt_x, dtype=np.float64)
    valid = np.isfinite(p) & np.isfinite(g)
    idx = np.where(valid)[0]
    correct = 0
    total = 0
    for a_pos in range(len(idx)):
        for b_pos in range(a_pos + 1, len(idx)):
            a, b = idx[a_pos], idx[b_pos]
            sep = abs(float(g[a] - g[b]))
            if max_gt_separation_m is not None and sep >= max_gt_separation_m:
                continue
            if sep < 1e-9:
                continue
            total += 1
            correct += int(np.sign(p[a] - p[b]) == np.sign(g[a] - g[b]))
    return float(correct / total) if total else float("nan")


def uncertainty_interval_coverage(
    gt_xyz: np.ndarray,
    q025_xyz: np.ndarray,
    q975_xyz: np.ndarray,
) -> Dict[str, float]:
    gt = np.asarray(gt_xyz, dtype=np.float64)
    lo = np.asarray(q025_xyz, dtype=np.float64)
    hi = np.asarray(q975_xyz, dtype=np.float64)
    mask = np.isfinite(gt) & np.isfinite(lo) & np.isfinite(hi)
    result = {}
    for axis, name in enumerate(("x", "y", "z")):
        m = mask[..., axis]
        if np.any(m):
            covered = (gt[..., axis][m] >= lo[..., axis][m]) & (gt[..., axis][m] <= hi[..., axis][m])
            result[f"coverage_{name}"] = float(np.mean(covered))
        else:
            result[f"coverage_{name}"] = float("nan")
    return result
