from __future__ import annotations

from typing import Dict, Mapping, Sequence
import numpy as np

H36M_GROUPS = {
    "head": (9, 10),
    "shoulder": (11, 14),
    "elbow": (12, 15),
    "wrist": (13, 16),
    "body": (0, 7, 8),
    "hip": (1, 4),
    "knee": (2, 5),
    "ankle": (3, 6),
}


def _as_pose_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"Expected (N,K,2) or (K,2), got {arr.shape}")
    return arr


def h36m_torso_scale(gt: np.ndarray) -> np.ndarray:
    """Shoulder-centre to hip-centre distance used by the 3DSP PDJ protocol."""
    gt = _as_pose_array(gt)
    hip = np.nanmean(gt[:, [1, 4], :], axis=1)
    shoulder = np.nanmean(gt[:, [11, 14], :], axis=1)
    scale = np.linalg.norm(shoulder - hip, axis=1)
    scale[~np.isfinite(scale) | (scale <= 1e-9)] = np.nan
    return scale


def normalized_joint_errors(pred, gt, *, scale: np.ndarray | None = None) -> np.ndarray:
    pred, gt = _as_pose_array(pred), _as_pose_array(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    if scale is None:
        if gt.shape[1] != 17:
            raise ValueError("Automatic torso scale is defined for H36M17 only")
        scale = h36m_torso_scale(gt)
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if len(scale) != len(gt):
        raise ValueError("scale length mismatch")
    err = np.linalg.norm(pred - gt, axis=2)
    err /= scale[:, None]
    valid = np.isfinite(pred).all(axis=2) & np.isfinite(gt).all(axis=2) & np.isfinite(scale)[:, None]
    err[~valid] = np.nan
    return err


def pdj_from_errors(errors: np.ndarray, threshold: float = 0.5, indices: Sequence[int] | None = None) -> float:
    e = np.asarray(errors, dtype=float)
    if indices is not None:
        e = e[:, np.asarray(indices, dtype=int)]
    valid = np.isfinite(e)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(e[valid] <= float(threshold)))


def pdj_auc_from_errors(errors: np.ndarray, *, max_threshold: float = 0.5, steps: int = 101,
                        indices: Sequence[int] | None = None) -> float:
    thresholds = np.linspace(0.0, float(max_threshold), int(steps))
    vals = np.asarray([pdj_from_errors(errors, t, indices) for t in thresholds], dtype=float)
    valid = np.isfinite(vals)
    if np.sum(valid) < 2:
        return float("nan")
    # Normalize by max_threshold so perfect predictions approach 1.0.
    x = thresholds[valid]
    y = vals[valid]
    area = np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return float(area / float(max_threshold))


def summarize_pdj(pred, gt, *, threshold: float = 0.5, max_auc_threshold: float = 0.5) -> Dict[str, object]:
    errors = normalized_joint_errors(pred, gt)
    joint_pdj = [pdj_from_errors(errors, threshold, [i]) for i in range(errors.shape[1])]
    groups = {
        name: {
            "PDJ": pdj_from_errors(errors, threshold, idxs),
            "AUC": pdj_auc_from_errors(errors, max_threshold=max_auc_threshold, indices=idxs),
        }
        for name, idxs in H36M_GROUPS.items()
    }
    finite = errors[np.isfinite(errors)]
    return {
        "PDJ": pdj_from_errors(errors, threshold),
        "AUC": pdj_auc_from_errors(errors, max_threshold=max_auc_threshold),
        "threshold": float(threshold),
        "auc_max_threshold": float(max_auc_threshold),
        "mean_normalized_error": None if finite.size == 0 else float(np.mean(finite)),
        "median_normalized_error": None if finite.size == 0 else float(np.median(finite)),
        "joint_PDJ": joint_pdj,
        "groups": groups,
        "valid_joint_observations": int(np.isfinite(errors).sum()),
    }


def bbox_scale_xyxy(bboxes) -> np.ndarray:
    b = np.asarray(bboxes, dtype=float)
    if b.ndim == 1:
        b = b[None]
    if b.shape[1] != 4:
        raise ValueError("Expected Nx4 bboxes")
    w = np.maximum(b[:, 2] - b[:, 0], 1e-9)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-9)
    return np.maximum(w, h)


def summarize_pck(pred, gt, bboxes_xyxy, *, thresholds=(0.05, 0.10), groups: Mapping[str, Sequence[int]] | None = None):
    pred, gt = _as_pose_array(pred), _as_pose_array(gt)
    if pred.shape != gt.shape:
        raise ValueError("Prediction/GT shape mismatch")
    scale = bbox_scale_xyxy(bboxes_xyxy)
    errors = normalized_joint_errors(pred, gt, scale=scale)
    result: Dict[str, object] = {}
    for t in thresholds:
        result[f"PCK@{float(t):.2f}"] = pdj_from_errors(errors, float(t))
    finite = errors[np.isfinite(errors)]
    result["median_normalized_error"] = None if finite.size == 0 else float(np.median(finite))
    if groups:
        result["groups"] = {
            name: {f"PCK@{float(t):.2f}": pdj_from_errors(errors, float(t), idxs) for t in thresholds}
            for name, idxs in groups.items()
        }
    return result
