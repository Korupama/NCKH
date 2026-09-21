from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..consolidation import box_iou, ROLE_BY_LABEL_ID

ROLES = ("player", "goalkeeper", "referee", "other")
CANDIDATE_ROLES = {"player", "goalkeeper"}


def _hungarian(pred: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], threshold: float):
    if not pred or not gt:
        return [], list(range(len(pred))), list(range(len(gt)))
    cost = np.ones((len(gt), len(pred)), dtype=float)
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            cost[gi, pi] = 1.0 - box_iou(g["bbox_xyxy"], p["bbox_xyxy"])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    mp, mg = set(), set()
    for gi, pi in zip(rows.tolist(), cols.tolist()):
        iou = 1.0 - float(cost[gi, pi])
        if iou >= threshold:
            matches.append((pi, gi, iou))
            mp.add(pi); mg.add(gi)
    return matches, [i for i in range(len(pred)) if i not in mp], [i for i in range(len(gt)) if i not in mg]


def _pred_role(p: Mapping[str, Any]) -> str:
    return str(p.get("role", p.get("resolved_role", "other")))


def _pred_candidate(p: Mapping[str, Any]) -> bool:
    return bool(p.get("candidate_for_stage3", p.get("candidate_hint", _pred_role(p) in CANDIDATE_ROLES)))


def evaluate_selected_frame(
    pred: Sequence[Mapping[str, Any]],
    gt: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    matches, fp, fn = _hungarian(pred, gt, iou_threshold)
    by_gt = {gi: pi for pi, gi, _ in matches}

    candidate_gt = [i for i, g in enumerate(gt) if str(g.get("role")) in CANDIDATE_ROLES]
    candidate_pred = [i for i, p in enumerate(pred) if _pred_candidate(p)]
    referee_gt = [i for i, g in enumerate(gt) if str(g.get("role")) == "referee"]
    # Candidate metrics are matched in candidate-only space so a referee/staff candidate
    # cannot be counted as a successful footballer merely because the all-human matcher
    # found a nearby non-footballer GT.
    cand_matches, cand_fp_local, cand_fn_local = _hungarian(
        [pred[i] for i in candidate_pred], [gt[i] for i in candidate_gt], iou_threshold
    )
    candidate_hits = len(cand_matches)
    candidate_fp = len(cand_fp_local)
    candidate_fn = len(cand_fn_local)
    referee_leaks = sum(gi in by_gt and _pred_candidate(pred[by_gt[gi]]) for gi in referee_gt)

    confusion = {r: {p: 0 for p in ROLES} for r in ROLES}
    role_correct = 0
    for pi, gi, _ in matches:
        gr = str(gt[gi].get("role", "other"))
        pr = _pred_role(pred[pi])
        if gr not in confusion: gr = "other"
        if pr not in ROLES: pr = "other"
        confusion[gr][pr] += 1
        role_correct += int(gr == pr)

    # Role precision/recall includes unmatched predictions/GT as role FP/FN.
    role_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ROLES}
    for r in ROLES:
        role_counts[r]["tp"] = confusion[r][r]
        role_counts[r]["fn"] = sum(confusion[r][p] for p in ROLES if p != r) + sum(1 for gi in fn if str(gt[gi].get("role", "other")) == r)
        role_counts[r]["fp"] = sum(confusion[g][r] for g in ROLES if g != r) + sum(1 for pi in fp if _pred_role(pred[pi]) == r)
    role_metrics = {}
    f1_supported = []
    for r, c in role_counts.items():
        precision = c["tp"] / max(1, c["tp"] + c["fp"])
        recall = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall) if precision + recall else 0.0
        support = c["tp"] + c["fn"]
        role_metrics[r] = {**c, "support": support, "precision": precision, "recall": recall, "f1": f1}
        if support > 0:
            f1_supported.append(f1)

    return {
        "iou_threshold": iou_threshold,
        "tp": len(matches), "fp": len(fp), "fn": len(fn),
        "precision": len(matches) / max(1, len(matches) + len(fp)),
        "recall": len(matches) / max(1, len(matches) + len(fn)),
        "role_accuracy_on_matched": role_correct / max(1, len(matches)),
        "role_macro_f1_supported": float(np.mean(f1_supported)) if f1_supported else 0.0,
        "role_metrics": role_metrics,
        "confusion": confusion,
        "candidate_gt": len(candidate_gt),
        "candidate_pred": len(candidate_pred),
        "candidate_hits": candidate_hits,
        "candidate_fp": candidate_fp,
        "candidate_fn": candidate_fn,
        "CandidatePrecision": candidate_hits / max(1, candidate_hits + candidate_fp),
        "CandidateRecall": candidate_hits / max(1, len(candidate_gt)),
        "referee_gt": len(referee_gt),
        "referee_leaks": referee_leaks,
        "RefereeLeakageRate": referee_leaks / max(1, len(referee_gt)),
        "matches": [{"pred_index": pi, "gt_index": gi, "iou": iou} for pi, gi, iou in matches],
    }


def _matched_records(records: Sequence[Mapping[str, Any]], g: Mapping[str, Any], match_iou: float) -> List[Mapping[str, Any]]:
    return [r for r in records if box_iou(r["bbox_xyxy"], g["bbox_xyxy"]) >= match_iou]


def gt_duplicate_diagnostics(
    records: Sequence[Mapping[str, Any]],
    gt: Sequence[Mapping[str, Any]],
    *,
    match_iou: float = 0.5,
    consolidated: bool = False,
) -> Dict[str, Any]:
    """Comparable duplicate diagnostics before/after physical-human consolidation.

    * AnyDuplicateRate: >1 hypothesis matches the same GT human.
    * CrossClassConflictRate: >1 hypotheses match and their fine-class evidence spans
      at least two SST human classes. A single successfully merged hypothesis containing
      Player+GK evidence is therefore *not* a residual conflict.
    """
    if consolidated:
        human_records = list(records)
    else:
        human_records = [r for r in records if int(r.get("label_id", -1)) in ROLE_BY_LABEL_ID]
    any_dup = 0
    cross_conflict = 0
    examples = []
    for gi, g in enumerate(gt):
        matched = _matched_records(human_records, g, match_iou)
        if len(matched) > 1:
            any_dup += 1
        fine_classes = set()
        if consolidated:
            for r in matched:
                evidence = r.get("class_evidence") or {}
                fine_classes.update(str(k) for k, v in evidence.items() if float(v) > 0)
                if not evidence and r.get("representative_label") is not None:
                    fine_classes.add(str(r.get("representative_label")))
        else:
            fine_classes.update(str(int(r.get("label_id", -1))) for r in matched)
        residual_cross = len(matched) > 1 and len(fine_classes) >= 2
        if residual_cross:
            cross_conflict += 1
        if len(matched) > 1:
            examples.append({
                "gt_index": gi,
                "gt_role": g.get("role"),
                "matched_hypotheses": len(matched),
                "fine_classes": sorted(fine_classes),
                "source_detection_ids": [
                    r.get("detection_id", r.get("physical_human_id")) for r in matched
                ],
                "cross_class_conflict": residual_cross,
            })
    den = max(1, len(gt))
    return {
        "gt_humans": len(gt),
        "any_duplicate_gt_humans": any_dup,
        "cross_class_conflict_gt_humans": cross_conflict,
        "AnyDuplicateRate": any_dup / den,
        "CrossClassConflictRate": cross_conflict / den,
        "examples": examples,
    }


def gt_cross_class_duplicate_rate(raw_detections: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], *, match_iou: float = 0.5) -> Dict[str, Any]:
    out = gt_duplicate_diagnostics(raw_detections, gt, match_iou=match_iou, consolidated=False)
    # Legacy keys retained for v1.2 report readers.
    return {
        **out,
        "duplicate_gt_humans": out["cross_class_conflict_gt_humans"],
        "CrossClassDuplicateRate": out["CrossClassConflictRate"],
    }


def gt_post_consolidation_duplicate_rate(humans: Sequence[Mapping[str, Any]], gt: Sequence[Mapping[str, Any]], *, match_iou: float = 0.5) -> Dict[str, Any]:
    out = gt_duplicate_diagnostics(humans, gt, match_iou=match_iou, consolidated=True)
    return {
        **out,
        "duplicate_gt_humans": out["any_duplicate_gt_humans"],
        "PostConsolidationDuplicateRate": out["AnyDuplicateRate"],
    }


def _ap101(rec: np.ndarray, prec: np.ndarray) -> float:
    if rec.size == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 101)
    vals = []
    for r in levels:
        mask = rec >= r
        vals.append(float(prec[mask].max()) if np.any(mask) else 0.0)
    return float(np.mean(vals))


def detection_ap(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    iou_thresholds: Sequence[float] = tuple(np.arange(0.50, 0.96, 0.05)),
    role: str | None = None,
) -> Dict[str, Any]:
    """COCO-style 101-point AP on benchmark target frames.

    Records must contain `frame_key`, `bbox_xyxy`; predictions also contain `score`.
    If role is supplied, both GT and prediction are filtered to that final role.
    """
    if role is not None:
        predictions = [x for x in predictions if str(x.get("role")) == role]
        ground_truth = [x for x in ground_truth if str(x.get("role")) == role]
    gt_by_frame: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for g in ground_truth:
        gt_by_frame[str(g["frame_key"])].append(g)
    pred_sorted = sorted(predictions, key=lambda x: -float(x.get("score", 0.0)))
    aps = []
    details = {}
    for thr in iou_thresholds:
        used = {k: set() for k in gt_by_frame}
        tp = np.zeros(len(pred_sorted), dtype=float)
        fp = np.zeros(len(pred_sorted), dtype=float)
        for i, p in enumerate(pred_sorted):
            key = str(p["frame_key"])
            candidates = gt_by_frame.get(key, [])
            best_iou = -1.0; best_j = None
            for j, g in enumerate(candidates):
                if j in used.get(key, set()):
                    continue
                iou = box_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= float(thr):
                tp[i] = 1.0; used[key].add(best_j)
            else:
                fp[i] = 1.0
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / max(1, len(ground_truth))
        prec = tp_c / np.maximum(1.0, tp_c + fp_c)
        ap = _ap101(rec, prec)
        aps.append(ap)
        details[f"AP@{thr:.2f}"] = ap
    return {
        "role": role or "all_humans",
        "num_gt": len(ground_truth),
        "num_pred": len(pred_sorted),
        "AP50": float(aps[0]) if aps else 0.0,
        "mAP50_95": float(np.mean(aps)) if aps else 0.0,
        "per_threshold": details,
    }


@dataclass
class DetectionAccumulator:
    selected_frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    candidate_gt: int = 0
    candidate_pred: int = 0
    candidate_hits: int = 0
    candidate_fp: int = 0
    candidate_fn: int = 0
    referee_gt: int = 0
    referee_leaks: int = 0
    role_counts: Dict[str, Dict[str, int]] = field(default_factory=lambda: {r: {"tp": 0, "fp": 0, "fn": 0} for r in ROLES})
    confusion: Dict[str, Dict[str, int]] = field(default_factory=lambda: {r: {p: 0 for p in ROLES} for r in ROLES})
    raw_any_dup_num: int = 0
    raw_cross_dup_num: int = 0
    raw_dup_den: int = 0
    post_any_dup_num: int = 0
    post_cross_dup_num: int = 0
    post_dup_den: int = 0

    def add(self, result: Mapping[str, Any], raw_dup: Mapping[str, Any], post_dup: Mapping[str, Any]) -> None:
        self.selected_frames += 1
        self.tp += int(result["tp"]); self.fp += int(result["fp"]); self.fn += int(result["fn"])
        self.candidate_gt += int(result["candidate_gt"]); self.candidate_pred += int(result.get("candidate_pred", 0))
        self.candidate_hits += int(result["candidate_hits"]); self.candidate_fp += int(result.get("candidate_fp", 0)); self.candidate_fn += int(result.get("candidate_fn", 0))
        self.referee_gt += int(result["referee_gt"]); self.referee_leaks += int(result["referee_leaks"])
        for r in ROLES:
            for k in ("tp", "fp", "fn"):
                self.role_counts[r][k] += int(result["role_metrics"][r][k])
            for p in ROLES:
                self.confusion[r][p] += int(result["confusion"][r][p])
        self.raw_any_dup_num += int(raw_dup.get("any_duplicate_gt_humans", raw_dup.get("duplicate_gt_humans", 0)))
        self.raw_cross_dup_num += int(raw_dup.get("cross_class_conflict_gt_humans", raw_dup.get("duplicate_gt_humans", 0)))
        self.raw_dup_den += int(raw_dup["gt_humans"])
        self.post_any_dup_num += int(post_dup.get("any_duplicate_gt_humans", post_dup.get("duplicate_gt_humans", 0)))
        self.post_cross_dup_num += int(post_dup.get("cross_class_conflict_gt_humans", 0))
        self.post_dup_den += int(post_dup["gt_humans"])

    def summary(self) -> Dict[str, Any]:
        role_metrics = {}
        f1s = []
        for r, c in self.role_counts.items():
            p = c["tp"] / max(1, c["tp"] + c["fp"])
            re = c["tp"] / max(1, c["tp"] + c["fn"])
            f1 = 2*p*re/max(1e-12,p+re) if p+re else 0.0
            support = c["tp"] + c["fn"]
            role_metrics[r] = {**c, "support": support, "precision": p, "recall": re, "f1": f1}
            if support > 0: f1s.append(f1)
        return {
            "selected_frames": self.selected_frames,
            "precision": self.tp / max(1, self.tp + self.fp),
            "recall": self.tp / max(1, self.tp + self.fn),
            "CandidatePrecision": self.candidate_hits / max(1, self.candidate_hits + self.candidate_fp),
            "CandidateRecall": self.candidate_hits / max(1, self.candidate_gt),
            "RefereeLeakageRate": self.referee_leaks / max(1, self.referee_gt),
            "AnyDuplicateRate_before": self.raw_any_dup_num / max(1, self.raw_dup_den),
            "AnyDuplicateRate_after": self.post_any_dup_num / max(1, self.post_dup_den),
            "CrossClassConflictRate_before": self.raw_cross_dup_num / max(1, self.raw_dup_den),
            "CrossClassConflictRate_after": self.post_cross_dup_num / max(1, self.post_dup_den),
            # Legacy aliases now compare the same cross-class-conflict semantic.
            "CrossClassDuplicateRate_before": self.raw_cross_dup_num / max(1, self.raw_dup_den),
            "CrossClassDuplicateRate_after": self.post_cross_dup_num / max(1, self.post_dup_den),
            "role_macro_f1_supported": float(np.mean(f1s)) if f1s else 0.0,
            "role_metrics": role_metrics,
            "confusion": self.confusion,
            "counts": {
                "tp": self.tp, "fp": self.fp, "fn": self.fn,
                "candidate_gt": self.candidate_gt, "candidate_pred": self.candidate_pred, "candidate_hits": self.candidate_hits,
                "candidate_fp": self.candidate_fp, "candidate_fn": self.candidate_fn,
                "referee_gt": self.referee_gt, "referee_leaks": self.referee_leaks,
                "raw_any_duplicate_gt": self.raw_any_dup_num, "raw_cross_class_conflict_gt": self.raw_cross_dup_num,
                "post_any_duplicate_gt": self.post_any_dup_num, "post_cross_class_conflict_gt": self.post_cross_dup_num,
            },
        }
