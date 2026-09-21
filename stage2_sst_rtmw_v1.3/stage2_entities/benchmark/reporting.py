from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import csv
import json

import numpy as np

ROLES=("player","goalkeeper","referee","other")


def write_json(path: str | Path, payload: Any) -> Path:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    return p


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return p
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with p.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return p


def flatten_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    out={}
    det=summary.get("detection",{})
    tr=summary.get("tracking_candidate",{})
    th=summary.get("tracking_human",{})
    ap=summary.get("detection_ap",{})
    for k in (
        "precision","recall","CandidatePrecision","CandidateRecall","RefereeLeakageRate",
        "AnyDuplicateRate_before","AnyDuplicateRate_after",
        "CrossClassConflictRate_before","CrossClassConflictRate_after",
        "CrossClassDuplicateRate_before","CrossClassDuplicateRate_after",
        "role_macro_f1_supported"
    ):
        out[k]=det.get(k)
    role_metrics = det.get("role_metrics", {})
    for role in ROLES:
        rm = role_metrics.get(role, {})
        out[f"{role}_precision"] = rm.get("precision")
        out[f"{role}_recall"] = rm.get("recall")
        out[f"{role}_f1"] = rm.get("f1")
    for prefix,src in (("candidate",tr),("human",th)):
        for k in ("HOTA","DetA","AssA","IDF1","IDR","IDP","TCR","AnchorCoverage","ConditionalTCR","IDSW","Fragmentations"):
            out[f"{prefix}_{k}"]=src.get(k)
    if "all_humans" in ap:
        out["human_AP50"]=ap["all_humans"].get("AP50")
        out["human_mAP50_95"]=ap["all_humans"].get("mAP50_95")
    for role in ROLES:
        if role in ap:
            out[f"{role}_AP50"] = ap[role].get("AP50")
            out[f"{role}_mAP50_95"] = ap[role].get("mAP50_95")
    runtime = summary.get("runtime", {})
    for k in ("sst_ms_per_frame","rtmw_ms_per_frame","perception_total_ms_per_frame","tracking_ms_per_window_primary"):
        out[k] = runtime.get(k)
    out["acceptance_gate_passed"] = summary.get("acceptance_gate", {}).get("passed")
    return out


def write_confusion_matrix(path: str | Path, confusion: Mapping[str, Mapping[str, int]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    mat=np.asarray([[int(confusion.get(g,{}).get(p,0)) for p in ROLES] for g in ROLES],dtype=int)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(mat,cmap="Blues")
    ax.set_xticks(range(len(ROLES)),ROLES,rotation=35,ha="right")
    ax.set_yticks(range(len(ROLES)),ROLES)
    ax.set_xlabel("Predicted track-level role"); ax.set_ylabel("Ground-truth role")
    ax.set_title("Stage 2 role confusion matrix")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,str(mat[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.tight_layout()
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=160); plt.close(fig)
    return p


def write_failure_overlay(image_path: str | Path, gt: Sequence[Mapping[str, Any]], pred: Sequence[Mapping[str, Any]], output_path: str | Path, title: str = "") -> Path | None:
    try:
        import cv2
    except Exception:
        return None
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None: return None
    # GT: green, Prediction: red. This is benchmark/debug visual, not the Stage-2 user visual contract.
    for g in gt:
        x1,y1,x2,y2=[int(round(float(v))) for v in g["bbox_xyxy"]]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,0),2)
        cv2.putText(image,f"GT {g.get('role','?')} {g.get('track_id','')}",(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,220,0),1,cv2.LINE_AA)
    for p in pred:
        x1,y1,x2,y2=[int(round(float(v))) for v in p["bbox_xyxy"]]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,240),2)
        cv2.putText(image,f"PR {p.get('role','?')} {p.get('track_id','')}",(x1,min(image.shape[0]-4,y2+15)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,240),1,cv2.LINE_AA)
    if title:
        cv2.putText(image,title,(15,26),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(p),image); return p
