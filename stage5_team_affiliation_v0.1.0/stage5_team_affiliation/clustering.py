from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from sklearn.cluster import KMeans

from .config import Stage5Config


@dataclass
class ClusterResult:
    labels: Dict[str, int]
    status: Dict[str, str]
    distances: Dict[str, List[float]]
    margins: Dict[str, float]
    centroids_raw: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    diagnostics: Dict[str, Any]


def _mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def fit_dominant_team_prototypes(track_features, config, *, random_state=23, n_init=20):
    """Trim within each cluster, refit cores, then classify ALL human tracks.

    No role/team labels are accepted. Degenerate or singleton groups fail closed.
    Features are L2-normalized, not standardized dimension by dimension.
    """
    config.validate()
    tids = sorted(track_features)
    if len(tids) < 2 * config.min_core_tracks:
        raise ValueError('Insufficient tracks for two dominant cores')
    X = np.asarray([track_features[t] for t in tids], dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) <= 0):
        raise ValueError('Invalid appearance features')
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    if len(np.unique(np.round(X, 6), axis=0)) < 2:
        raise ValueError('Only one distinguishable appearance group')
    centers = KMeans(n_clusters=2, n_init=n_init, random_state=random_state).fit(X).cluster_centers_
    for _ in range(config.trim_iterations):
        distances = np.linalg.norm(X[:, None] - centers, axis=2)
        labels = distances.argmin(axis=1)
        cores = []
        for team in (0, 1):
            members = np.flatnonzero(labels == team)
            if len(members) < config.min_core_tracks:
                raise ValueError('Singleton/insufficient dominant team core')
            keep = max(config.min_core_tracks, int(np.ceil(len(members) * config.core_keep_fraction)))
            cores.append(members[np.argsort(distances[members, team], kind='stable')[:keep]])
        centers = np.stack([X[ids].mean(axis=0) for ids in cores])
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    if separation < config.min_centroid_separation:
        raise ValueError('Dominant prototypes insufficiently separated')
    # Rounding the 80% quota can retain a lone outlier in a small cluster.
    # Enforce the same separation cap for prototypes as for final membership.
    core_fallback_teams = []
    for team, ids in enumerate(cores):
        core_distances = np.linalg.norm(X[ids] - centers[team], axis=1)
        accepted = ids[core_distances <= separation * config.radius_separation_fraction]
        if len(accepted) < config.min_core_tracks:
            # The hard separation cap can reject every member of a small but
            # otherwise stable cluster (the TRAIN example is SNGS-160).  Keep
            # the nearest pre-cap members only as a deterministic last-resort
            # prototype core.  We still enforce two-cluster occupancy and the
            # refined centroid-separation gate below, so this does not turn a
            # degenerate one-colour sequence into a successful fit.
            order = np.argsort(core_distances, kind='stable')
            accepted = ids[order[:config.min_core_tracks]]
            core_fallback_teams.append(team)
        cores[team] = accepted
    centers = np.stack([X[ids].mean(axis=0) for ids in cores])
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    if separation < config.min_centroid_separation:
        raise ValueError('Refined prototypes insufficiently separated')
    radii = []
    for team, ids in enumerate(cores):
        d = np.linalg.norm(X[ids] - centers[team], axis=1)
        radius = max(config.radius_floor, float(np.quantile(d, .95)) + 3 * _mad(d))
        radii.append(min(radius, separation * config.radius_separation_fraction))
    result = {'centroids': centers.tolist(), 'radii': radii, 'separation': separation,
              'cores': [[tids[i] for i in ids] for ids in cores],
              'core_fallback_used': bool(core_fallback_teams),
              'core_fallback_teams': core_fallback_teams,
              'core_fallback_reason': ('MINIMUM_CORE_NEAREST_CENTROID'
                                       if core_fallback_teams else None)}
    result['tracks'] = {tid: classify_team_or_residual(X[i], result, config) for i, tid in enumerate(tids)}
    return result


def classify_team_or_residual(feature, prototypes, config):
    x = np.asarray(feature, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.linalg.norm(x) <= 0:
        return {'appearance_group': 'UNKNOWN', 'team_id': None}
    x = x / np.linalg.norm(x)
    d = np.linalg.norm(np.asarray(prototypes['centroids']) - x, axis=1)
    order = np.argsort(d); best, second = map(int, order)
    margin = float((d[second] - d[best]) / max(d[second], 1e-9))
    far_both = bool(np.all(d > np.asarray(prototypes['radii'])))
    accepted = d[best] <= prototypes['radii'][best] and margin >= config.min_team_margin
    return {'appearance_group': f'TEAM_{best}' if accepted else 'RESIDUAL' if far_both else 'AMBIGUOUS',
            'team_id': best if accepted else None, 'distances': d.tolist(), 'margin': margin}


def fit_two_teams(track_features: Mapping[str, np.ndarray], config: Stage5Config) -> ClusterResult:
    tids = sorted(track_features)
    if len(tids) < 4:
        raise ValueError(f"Need at least 4 outfield tracks for two-team clustering; got {len(tids)}")
    X = np.stack([track_features[t] for t in tids], axis=0).astype(np.float64)
    if np.unique(np.round(X, 6), axis=0).shape[0] < 2:
        raise ValueError("Outfield appearance features do not contain two distinguishable groups")
    # Features are already L2-normalized. KMeans is intentionally fit in the
    # original feature space; per-dimension standardization makes sparse colour
    # histogram bins with tiny variance dominate the distance metric.
    Xs = X
    km = KMeans(n_clusters=2, n_init=config.kmeans_n_init, random_state=config.random_state)
    y = km.fit_predict(Xs)
    d = km.transform(Xs)

    assigned = d[np.arange(len(tids)), y]
    thresholds: Dict[int, float] = {}
    for c in (0, 1):
        vals = assigned[y == c]
        if len(vals) == 0:
            thresholds[c] = float("inf")
        else:
            med = float(np.median(vals))
            mad = _mad(vals)
            thresholds[c] = med + config.outlier_mad_factor * max(mad, 0.05)

    labels: Dict[str, int] = {}
    status: Dict[str, str] = {}
    distances: Dict[str, List[float]] = {}
    margins: Dict[str, float] = {}
    for i, tid in enumerate(tids):
        c = int(y[i])
        labels[tid] = c
        distances[tid] = [float(d[i, 0]), float(d[i, 1])]
        a, b = sorted([float(d[i, 0]), float(d[i, 1])])
        margin = float((b - a) / max(b, 1e-9))
        margins[tid] = margin
        outlier = float(assigned[i]) > thresholds[c]
        low_margin = margin < config.min_cluster_margin
        status[tid] = "UNKNOWN" if (outlier or low_margin) else "VALID"

    centroids_raw = km.cluster_centers_.copy()
    return ClusterResult(
        labels=labels,
        status=status,
        distances=distances,
        margins=margins,
        centroids_raw=centroids_raw,
        scaler_mean=np.zeros(X.shape[1], dtype=np.float64),
        scaler_scale=np.ones(X.shape[1], dtype=np.float64),
        diagnostics={
            "inertia": float(km.inertia_),
            "cluster_sizes": {str(c): int(np.sum(y == c)) for c in (0, 1)},
            "distance_thresholds": {str(k): float(v) for k, v in thresholds.items()},
        },
    )


def assign_to_centroids(feature: np.ndarray, centroids: np.ndarray, min_margin: float = 0.1, max_distance_ratio: float = 1.35) -> Dict[str, Any]:
    feature = np.asarray(feature, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    d = np.linalg.norm(centroids - feature[None, :], axis=1)
    order = np.argsort(d)
    best, second = int(order[0]), int(order[1])
    margin = float((d[second] - d[best]) / max(d[second], 1e-9))
    ratio = float(d[best] / max(d[second], 1e-9))
    valid = margin >= min_margin and ratio <= max_distance_ratio
    return {
        "team_id": best if valid else None,
        "status": "VALID" if valid else "UNKNOWN",
        "distances": [float(x) for x in d],
        "margin": margin,
        "distance_ratio": ratio,
    }
