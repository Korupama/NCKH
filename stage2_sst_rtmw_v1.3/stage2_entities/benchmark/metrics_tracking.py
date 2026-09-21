from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..consolidation import box_iou


def _group_by_frame(records: Sequence[Mapping[str, Any]]) -> Dict[int, List[Mapping[str, Any]]]:
    out: Dict[int, List[Mapping[str, Any]]] = {}
    for x in records:
        out.setdefault(int(x["frame_index"]), []).append(x)
    return out


def _tracking_data(gt_records: Sequence[Mapping[str, Any]], pred_records: Sequence[Mapping[str, Any]], frame_indices: Sequence[int]):
    gt_by = _group_by_frame(gt_records); pr_by = _group_by_frame(pred_records)
    gt_ids = sorted({str(x["track_id"]) for x in gt_records})
    pr_ids = sorted({str(x["track_id"]) for x in pred_records})
    gmap = {x:i for i,x in enumerate(gt_ids)}; pmap = {x:i for i,x in enumerate(pr_ids)}
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [], "num_gt_ids": len(gt_ids), "num_tracker_ids": len(pr_ids), "num_gt_dets": len(gt_records), "num_tracker_dets": len(pred_records)}
    for fi in frame_indices:
        gs = gt_by.get(int(fi), []); ps = pr_by.get(int(fi), [])
        ga = np.asarray([gmap[str(x["track_id"])] for x in gs], dtype=int)
        pa = np.asarray([pmap[str(x["track_id"])] for x in ps], dtype=int)
        sim = np.zeros((len(gs), len(ps)), dtype=float)
        for i,g in enumerate(gs):
            for j,p in enumerate(ps): sim[i,j] = box_iou(g["bbox_xyxy"], p["bbox_xyxy"])
        data["gt_ids"].append(ga); data["tracker_ids"].append(pa); data["similarity_scores"].append(sim)
    return data


def _hota_sequence(data: Mapping[str, Any], alphas: np.ndarray | None = None) -> Dict[str, Any]:
    """Self-contained HOTA implementation matching TrackEval's reference equations.

    Algorithm provenance: Jonathon Luiten et al. TrackEval/HOTA (MIT). We retain only
    the image-IoU tracking fields needed by Stage 2, without importing TrackEval.
    """
    if alphas is None:
        alphas = np.arange(0.05, 0.99, 0.05)
    n_a = len(alphas)
    tp = np.zeros(n_a); fn = np.zeros(n_a); fp = np.zeros(n_a); loca_sum = np.zeros(n_a)
    if data["num_tracker_dets"] == 0:
        fn[:] = data["num_gt_dets"]
        return _hota_final(tp, fn, fp, np.zeros(n_a), loca_sum, alphas)
    if data["num_gt_dets"] == 0:
        fp[:] = data["num_tracker_dets"]
        return _hota_final(tp, fn, fp, np.zeros(n_a), loca_sum, alphas)

    potential = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]), dtype=float)
    gt_count = np.zeros((data["num_gt_ids"], 1), dtype=float)
    pr_count = np.zeros((1, data["num_tracker_ids"]), dtype=float)
    eps = np.finfo(float).eps
    for gt_ids_t, pr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        if len(gt_ids_t) and len(pr_ids_t):
            denom = sim.sum(0)[None,:] + sim.sum(1)[:,None] - sim
            normalized = np.zeros_like(sim)
            mask = denom > eps
            normalized[mask] = sim[mask] / denom[mask]
            potential[gt_ids_t[:,None], pr_ids_t[None,:]] += normalized
        gt_count[gt_ids_t] += 1
        pr_count[0, pr_ids_t] += 1
    global_align = potential / np.maximum(eps, gt_count + pr_count - potential)
    match_counts = [np.zeros_like(potential) for _ in alphas]

    for gt_ids_t, pr_ids_t, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        if len(gt_ids_t) == 0:
            fp += len(pr_ids_t); continue
        if len(pr_ids_t) == 0:
            fn += len(gt_ids_t); continue
        score = global_align[gt_ids_t[:,None], pr_ids_t[None,:]] * sim
        rows, cols = linear_sum_assignment(-score)
        for ai, alpha in enumerate(alphas):
            mask = sim[rows, cols] >= float(alpha) - eps
            rr, cc = rows[mask], cols[mask]
            nm = len(rr)
            tp[ai] += nm; fn[ai] += len(gt_ids_t)-nm; fp[ai] += len(pr_ids_t)-nm
            if nm:
                loca_sum[ai] += float(sim[rr, cc].sum())
                match_counts[ai][gt_ids_t[rr], pr_ids_t[cc]] += 1

    assa = np.zeros(n_a)
    for ai in range(n_a):
        mc = match_counts[ai]
        ass_pair = mc / np.maximum(1.0, gt_count + pr_count - mc)
        assa[ai] = float((mc * ass_pair).sum() / max(1.0, tp[ai]))
    return _hota_final(tp, fn, fp, assa, loca_sum, alphas)


def _hota_final(tp, fn, fp, assa, loca_sum, alphas):
    deta = tp / np.maximum(1.0, tp + fn + fp)
    detre = tp / np.maximum(1.0, tp + fn)
    detpr = tp / np.maximum(1.0, tp + fp)
    loca = np.where(tp > 0, loca_sum / np.maximum(1e-12, tp), 1.0)
    hota = np.sqrt(deta * assa)
    return {
        "alphas": [float(x) for x in alphas],
        "HOTA_array": hota.tolist(), "DetA_array": deta.tolist(), "AssA_array": assa.tolist(), "LocA_array": loca.tolist(),
        "TP_array": tp.astype(int).tolist(), "FN_array": fn.astype(int).tolist(), "FP_array": fp.astype(int).tolist(),
        "HOTA": float(hota.mean()), "DetA": float(deta.mean()), "AssA": float(assa.mean()), "LocA": float(loca.mean()),
        "DetRe": float(detre.mean()), "DetPr": float(detpr.mean()),
    }


def _identity_sequence(data: Mapping[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    ng, npred = data["num_gt_ids"], data["num_tracker_ids"]
    if data["num_tracker_dets"] == 0:
        return {"IDTP": 0, "IDFN": int(data["num_gt_dets"]), "IDFP": 0, "IDF1": 0.0, "IDR": 0.0, "IDP": 0.0}
    if data["num_gt_dets"] == 0:
        return {"IDTP": 0, "IDFN": 0, "IDFP": int(data["num_tracker_dets"]), "IDF1": 0.0, "IDR": 0.0, "IDP": 0.0}
    potential = np.zeros((ng, npred)); gc = np.zeros(ng); pc = np.zeros(npred)
    for gids, pids, sim in zip(data["gt_ids"], data["tracker_ids"], data["similarity_scores"]):
        if len(gids) and len(pids):
            rr, cc = np.nonzero(sim >= threshold)
            potential[gids[rr], pids[cc]] += 1
        gc[gids] += 1; pc[pids] += 1
    size = ng + npred
    fp_mat = np.zeros((size, size)); fn_mat = np.zeros((size, size))
    fp_mat[ng:, :npred] = 1e10; fn_mat[:ng, npred:] = 1e10
    for g in range(ng):
        fn_mat[g, :npred] = gc[g]; fn_mat[g, npred+g] = gc[g]
    for p in range(npred):
        fp_mat[:ng, p] = pc[p]; fp_mat[ng+p, p] = pc[p]
    fn_mat[:ng,:npred] -= potential; fp_mat[:ng,:npred] -= potential
    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn = int(round(float(fn_mat[rows,cols].sum()))); idfp = int(round(float(fp_mat[rows,cols].sum())))
    idtp = int(round(float(gc.sum()))) - idfn
    idr = idtp / max(1, idtp + idfn); idp = idtp / max(1, idtp + idfp)
    idf1 = 2*idtp/max(1, 2*idtp+idfn+idfp)
    return {"IDTP": idtp, "IDFN": idfn, "IDFP": idfp, "IDF1": idf1, "IDR": idr, "IDP": idp}


def _frame_matches(gt_records, pred_records, frame_indices, threshold=0.5):
    gt_by = _group_by_frame(gt_records); pr_by = _group_by_frame(pred_records)
    out = {}
    for fi in frame_indices:
        gs = gt_by.get(fi, []); ps = pr_by.get(fi, [])
        if not gs or not ps:
            out[fi] = []; continue
        cost = np.ones((len(gs),len(ps)))
        for i,g in enumerate(gs):
            for j,p in enumerate(ps): cost[i,j] = 1-box_iou(g["bbox_xyxy"],p["bbox_xyxy"])
        rr,cc=linear_sum_assignment(cost)
        out[fi]=[(gs[i],ps[j],1-float(cost[i,j])) for i,j in zip(rr,cc) if 1-float(cost[i,j])>=threshold]
    return out


def id_switch_fragmentation(gt_records, pred_records, frame_indices, threshold=0.5) -> Dict[str, Any]:
    matches = _frame_matches(gt_records,pred_records,frame_indices,threshold)
    by_gt: Dict[str, List[Tuple[int,str]]] = {}
    visible: Dict[str, List[int]] = {}
    for g in gt_records: visible.setdefault(str(g["track_id"]), []).append(int(g["frame_index"]))
    for fi, ms in matches.items():
        for g,p,_ in ms: by_gt.setdefault(str(g["track_id"]),[]).append((fi,str(p["track_id"])))
    idsw=0; frag=0
    for gid, items in by_gt.items():
        items=sorted(items)
        last_pid=None
        for _,pid in items:
            if last_pid is not None and pid!=last_pid: idsw+=1
            last_pid=pid
        matched_frames={fi for fi,_ in items}; vis=sorted(set(visible.get(gid,[])))
        segments=0; active=False
        for fi in vis:
            hit=fi in matched_frames
            if hit and not active: segments+=1
            active=hit
        frag += max(0,segments-1)
    return {"IDSW": idsw, "Fragmentations": frag}


def evaluate_tcr(gt_records, pred_records, frame_indices, target_frame: int, threshold: float=0.5) -> Dict[str, Any]:
    gt_by=_group_by_frame(gt_records); pr_by=_group_by_frame(pred_records)
    gs=gt_by.get(target_frame,[]); ps=pr_by.get(target_frame,[])
    if not gs:
        return {
            "TCR":0.0,"anchors":0,"matched_anchors":0,"AnchorCoverage":0.0,
            "ConditionalTCR":0.0,"per_anchor":[]
        }
    # Anchor identities by t0 Hungarian matching. Missing anchors contribute 0 to overall TCR,
    # while ConditionalTCR isolates continuity after a track was successfully anchored.
    anchor_matches=_frame_matches(gt_records,pred_records,[target_frame],threshold).get(target_frame,[])
    pairs={str(g["track_id"]):str(p["track_id"]) for g,p,_ in anchor_matches}
    per=[]
    for g0 in gs:
        gid=str(g0["track_id"])
        visible=sum(any(str(x["track_id"])==gid for x in gt_by.get(fi,[])) for fi in frame_indices)
        if gid not in pairs:
            per.append({"gt_track_id":gid,"pred_track_id":None,"visible_frames":visible,"correct_frames":0,"TCR":0.0})
            continue
        pid=pairs[gid]; correct=0
        for fi in frame_indices:
            gitem=next((x for x in gt_by.get(fi,[]) if str(x["track_id"])==gid),None)
            if gitem is None: continue
            pitem=next((x for x in pr_by.get(fi,[]) if str(x["track_id"])==pid),None)
            if pitem is not None and box_iou(gitem["bbox_xyxy"],pitem["bbox_xyxy"])>=threshold:
                correct += 1
        per.append({"gt_track_id":gid,"pred_track_id":pid,"visible_frames":visible,"correct_frames":correct,"TCR":correct/max(1,visible)})
    matched=[x for x in per if x["pred_track_id"] is not None]
    return {
        "TCR":float(np.mean([x["TCR"] for x in per])) if per else 0.0,
        "anchors":len(per),
        "matched_anchors":len(matched),
        "AnchorCoverage":len(matched)/max(1,len(per)),
        "ConditionalTCR":float(np.mean([x["TCR"] for x in matched])) if matched else 0.0,
        "per_anchor":per,
    }


def evaluate_tracking_window(gt_records, pred_records, frame_indices, target_frame: int) -> Dict[str, Any]:
    data=_tracking_data(gt_records,pred_records,frame_indices)
    hota=_hota_sequence(data); identity=_identity_sequence(data)
    simple=id_switch_fragmentation(gt_records,pred_records,frame_indices)
    tcr=evaluate_tcr(gt_records,pred_records,frame_indices,target_frame)
    return {
        **hota, **identity, **simple,
        **{
            "TCR":tcr["TCR"],
            "TCR_anchors":tcr["anchors"],
            "TCR_matched_anchors":tcr["matched_anchors"],
            "AnchorCoverage":tcr["AnchorCoverage"],
            "ConditionalTCR":tcr["ConditionalTCR"],
            "TCR_details":tcr["per_anchor"],
        },
        "num_gt_dets":data["num_gt_dets"],"num_pred_dets":data["num_tracker_dets"],
        "num_gt_ids":data["num_gt_ids"],"num_pred_ids":data["num_tracker_ids"]
    }


def combine_tracking_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"HOTA":0.0,"DetA":0.0,"AssA":0.0,"IDF1":0.0,"TCR":0.0,"AnchorCoverage":0.0,"ConditionalTCR":0.0,"windows":0}
    tp=np.sum([np.asarray(r["TP_array"],dtype=float) for r in results],axis=0)
    fn=np.sum([np.asarray(r["FN_array"],dtype=float) for r in results],axis=0)
    fp=np.sum([np.asarray(r["FP_array"],dtype=float) for r in results],axis=0)
    ass_num=np.sum([np.asarray(r["AssA_array"],dtype=float)*np.asarray(r["TP_array"],dtype=float) for r in results],axis=0)
    loc_num=np.sum([np.asarray(r["LocA_array"],dtype=float)*np.asarray(r["TP_array"],dtype=float) for r in results],axis=0)
    assa=ass_num/np.maximum(1.0,tp); loca=np.where(tp>0,loc_num/np.maximum(1.0,tp),1.0)
    deta=tp/np.maximum(1.0,tp+fn+fp); hota=np.sqrt(deta*assa)
    idtp=sum(int(r["IDTP"]) for r in results); idfn=sum(int(r["IDFN"]) for r in results); idfp=sum(int(r["IDFP"]) for r in results)
    tcr_num=sum(float(r["TCR"])*int(r.get("TCR_anchors",0)) for r in results); tcr_den=sum(int(r.get("TCR_anchors",0)) for r in results)
    matched_den=sum(int(r.get("TCR_matched_anchors",0)) for r in results)
    conditional_num=sum(float(r.get("ConditionalTCR",0))*int(r.get("TCR_matched_anchors",0)) for r in results)
    return {
        "windows":len(results),"HOTA":float(hota.mean()),"DetA":float(deta.mean()),"AssA":float(assa.mean()),"LocA":float(loca.mean()),
        "IDTP":idtp,"IDFN":idfn,"IDFP":idfp,"IDF1":2*idtp/max(1,2*idtp+idfn+idfp),"IDR":idtp/max(1,idtp+idfn),"IDP":idtp/max(1,idtp+idfp),
        "IDSW":sum(int(r.get("IDSW",0)) for r in results),"Fragmentations":sum(int(r.get("Fragmentations",0)) for r in results),
        "TCR":tcr_num/max(1,tcr_den),"TCR_anchors":tcr_den,
        "TCR_matched_anchors":matched_den,
        "AnchorCoverage":matched_den/max(1,tcr_den),
        "ConditionalTCR":conditional_num/max(1,matched_den),
        "HOTA_array":hota.tolist(),"DetA_array":deta.tolist(),"AssA_array":assa.tolist(),"TP_array":tp.astype(int).tolist(),"FN_array":fn.astype(int).tolist(),"FP_array":fp.astype(int).tolist(),
    }
