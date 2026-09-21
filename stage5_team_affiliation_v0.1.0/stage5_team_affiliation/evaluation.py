from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, precision_recall_fscore_support


def _best_binary_mapping(gt: Sequence[int], pred: Sequence[int]) -> Dict[int, int]:
    gt = list(map(int, gt)); pred = list(map(int, pred))
    scores = {}
    for mapping in ({0: 0, 1: 1}, {0: 1, 1: 0}):
        correct = sum(int(mapping.get(p, -99) == g) for g, p in zip(gt, pred))
        scores[tuple(sorted(mapping.items()))] = correct
    best = max(scores, key=scores.get)
    return dict(best)


def evaluate_team_assignments(gt_by_track: Mapping[str, int], pred_by_track: Mapping[str, int | None]) -> Dict[str, Any]:
    tids = [t for t in gt_by_track if pred_by_track.get(t) in (0, 1)]
    unknown = [t for t in gt_by_track if pred_by_track.get(t) not in (0, 1)]
    if not tids:
        return {
            "judgeable_tracks": 0,
            "unknown_tracks": len(unknown),
            "unknown_rate": float(len(unknown) / max(1, len(gt_by_track))),
            "hungarian_matched_accuracy": None,
            "ARI": None,
            "NMI": None,
        }
    gt = [int(gt_by_track[t]) for t in tids]
    pred = [int(pred_by_track[t]) for t in tids]
    mapping = _best_binary_mapping(gt, pred)
    mapped = [mapping[p] for p in pred]
    acc = float(np.mean(np.asarray(mapped) == np.asarray(gt)))
    p, r, f1, support = precision_recall_fscore_support(gt, mapped, labels=[0,1], zero_division=0)
    return {
        "judgeable_tracks": len(tids),
        "unknown_tracks": len(unknown),
        "unknown_rate": float(len(unknown) / max(1, len(gt_by_track))),
        "mapping_pred_to_gt": {str(k): int(v) for k, v in mapping.items()},
        "hungarian_matched_accuracy": acc,
        "ARI": float(adjusted_rand_score(gt, pred)),
        "NMI": float(normalized_mutual_info_score(gt, pred)),
        "per_team": {
            str(i): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i in range(2)
        },
    }
