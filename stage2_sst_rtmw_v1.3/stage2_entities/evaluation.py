from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from scipy.optimize import linear_sum_assignment

from .consolidation import box_iou


def _match(pred: Sequence[Mapping[str,Any]], gt: Sequence[Mapping[str,Any]], threshold: float=0.5):
    if not pred or not gt:
        return [],list(range(len(pred))),list(range(len(gt)))
    cost=np.ones((len(gt),len(pred)),dtype=float)
    for gi,g in enumerate(gt):
        for pi,p in enumerate(pred): cost[gi,pi]=1-box_iou(g["bbox_xyxy"],p["bbox_xyxy"])
    gi,pi=linear_sum_assignment(cost)
    matches=[]; mp=set(); mg=set()
    for g,p in zip(gi,pi):
        iou=1-cost[g,p]
        if iou>=threshold:
            matches.append((p,g,float(iou)));mp.add(p);mg.add(g)
    return matches,[i for i in range(len(pred)) if i not in mp],[i for i in range(len(gt)) if i not in mg]


def selected_frame_metrics(pred: Sequence[Mapping[str,Any]], gt: Sequence[Mapping[str,Any]], iou_threshold: float=0.5) -> Dict[str,Any]:
    matches,fp,fn=_match(pred,gt,iou_threshold)
    candidate_roles={"player","goalkeeper"}
    candidate_pred=[p for p in pred if p.get("candidate_for_stage3",p.get("role") in candidate_roles)]
    candidate_gt_items=[g for g in gt if g.get("role") in candidate_roles]
    cand_matches,cand_fp,cand_fn=_match(candidate_pred,candidate_gt_items,iou_threshold)
    referee_gt=[i for i,g in enumerate(gt) if g.get("role")=="referee"]
    matched_by_gt={g:p for p,g,_ in matches}
    role_correct=0
    referee_leaks=0
    for pidx,gidx,_ in matches:
        if pred[pidx].get("role")==gt[gidx].get("role"): role_correct+=1
    for gidx in referee_gt:
        if gidx in matched_by_gt:
            p=pred[matched_by_gt[gidx]]
            if p.get("candidate_for_stage3",False): referee_leaks+=1
    return {
        "iou_threshold":iou_threshold,
        "precision":len(matches)/max(1,len(matches)+len(fp)),
        "recall":len(matches)/max(1,len(matches)+len(fn)),
        "role_accuracy_on_matched":role_correct/max(1,len(matches)),
        "CandidatePrecision":len(cand_matches)/max(1,len(cand_matches)+len(cand_fp)),
        "CandidateRecall":len(cand_matches)/max(1,len(candidate_gt_items)),
        "RefereeLeakageRate":referee_leaks/max(1,len(referee_gt)),
        "tp":len(matches),"fp":len(fp),"fn":len(fn),
        "candidate_tp":len(cand_matches),"candidate_fp":len(cand_fp),"candidate_fn":len(cand_fn),
    }


def raw_cross_class_duplicate_rate(raw_detections: Sequence[Mapping[str,Any]], *, iou_threshold: float=0.85) -> Dict[str,Any]:
    humans=[d for d in raw_detections if int(d.get("label_id",-1)) in {2,3,4,5,6}]
    involved=set(); pairs=[]
    for i in range(len(humans)):
        for j in range(i+1,len(humans)):
            if int(humans[i]["label_id"])==int(humans[j]["label_id"]): continue
            iou=box_iou(humans[i]["bbox_xyxy"],humans[j]["bbox_xyxy"])
            if iou>=iou_threshold:
                involved.update([i,j]); pairs.append({"a":humans[i]["detection_id"],"b":humans[j]["detection_id"],"iou":iou})
    return {
        "raw_human_detections":len(humans),
        "detections_in_cross_class_duplicate_pairs":len(involved),
        "CrossClassDuplicateDetectionRate":len(involved)/max(1,len(humans)),
        "pairs":pairs,
    }
