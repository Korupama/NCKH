from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import json
import numpy as np

from .metrics import summarize_pck

FOOT_GROUPS = {
    "left_foot": (17, 18, 19),
    "right_foot": (20, 21, 22),
    "toes": (17, 18, 20, 21),
    "heels": (19, 22),
    "ankles": (15, 16),
}


def _arr23(records) -> np.ndarray:
    arr = np.full((23, 2), np.nan, dtype=float)
    if isinstance(records, list):
        for i, v in enumerate(records[:23]):
            if isinstance(v, Mapping):
                x, y = v.get("x"), v.get("y")
            else:
                x, y = v[:2]
            if x is not None and y is not None:
                arr[i] = (float(x), float(y))
    elif isinstance(records, Mapping):
        for k, v in records.items():
            i = int(k)
            if 0 <= i < 23 and isinstance(v, Mapping):
                arr[i] = (float(v["x"]), float(v["y"]))
    return arr


def evaluate_pose23(gt_path: str | Path, pred_path: str | Path) -> Dict[str, Any]:
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    pred = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    gt_items = gt.get("annotations", gt if isinstance(gt, list) else [])
    pred_items = pred.get("predictions", pred if isinstance(pred, list) else [])
    pmap = {str(x["id"]): x for x in pred_items}
    gts, preds, boxes, ids = [], [], [], []
    for item in gt_items:
        key = str(item["id"])
        if key not in pmap:
            continue
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if bbox is None:
            continue
        if len(bbox) == 4 and item.get("bbox_format") == "xywh":
            x, y, w, h = map(float, bbox); bbox = [x, y, x+w, y+h]
        gts.append(_arr23(item.get("keypoints_23") or item.get("keypoints")))
        preds.append(_arr23(pmap[key].get("keypoints_23") or pmap[key].get("keypoints")))
        boxes.append(list(map(float, bbox)))
        ids.append(key)
    if not gts:
        raise RuntimeError("No matched Pose23 annotations")
    metrics = summarize_pck(np.stack(preds), np.stack(gts), np.asarray(boxes), groups=FOOT_GROUPS)
    return {"schema_version":"stage3-pose23-eval-1.0", "samples":len(gts), "metrics":metrics, "matched_ids":ids}
