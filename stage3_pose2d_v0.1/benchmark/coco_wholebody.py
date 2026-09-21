from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json
import cv2
import numpy as np

from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN

SIGMAS_BODY = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
SIGMAS_FOOT = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
SIGMAS_FACE = [0.042,0.043,0.044,0.043,0.040,0.035,0.031,0.025,0.020,0.023,0.029,0.032,0.037,0.038,0.043,0.041,0.045,0.013,0.012,0.011,0.011,0.012,0.012,0.011,0.011,0.013,0.015,0.009,0.007,0.007,0.007,0.012,0.009,0.008,0.016,0.010,0.017,0.011,0.009,0.011,0.009,0.007,0.013,0.008,0.011,0.012,0.010,0.034,0.008,0.008,0.009,0.008,0.008,0.007,0.010,0.008,0.009,0.009,0.009,0.007,0.007,0.008,0.011,0.008,0.008,0.008,0.010,0.008]
SIGMAS_HAND = [0.029,0.022,0.035,0.037,0.047,0.026,0.025,0.024,0.035,0.018,0.024,0.022,0.026,0.017,0.021,0.021,0.032,0.020,0.019,0.022,0.031]
SIGMAS_WHOLE = SIGMAS_BODY + SIGMAS_FOOT + SIGMAS_FACE + SIGMAS_HAND + SIGMAS_HAND


def _flat_kpts(xy: np.ndarray, scores: np.ndarray, indices: Sequence[int]) -> List[float]:
    out: List[float] = []
    for i in indices:
        x, y = xy[i]
        score = scores[i]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(score) and score > 0):
            out.extend([0.0, 0.0, 0.0])
        else:
            out.extend([float(x), float(y), float(score)])
    return out


def prediction_record(image_id: int, annotation_id: int, xy: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    if xy.shape != (133, 2) or scores.shape != (133,):
        raise ValueError("Expected WholeBody133 prediction")
    finite_scores = scores[np.isfinite(scores) & (scores > 0)]
    # Result ranking score only. Raw SimCC values are deliberately not interpreted as probabilities.
    ranking_score = 1.0 if finite_scores.size else 0.0
    return {
        "image_id": int(image_id),
        "category_id": 1,
        "annotation_id": int(annotation_id),
        "score": ranking_score,
        "keypoints": _flat_kpts(xy, scores, range(0, 17)),
        "foot_kpts": _flat_kpts(xy, scores, range(17, 23)),
        "face_kpts": _flat_kpts(xy, scores, range(23, 91)),
        "lefthand_kpts": _flat_kpts(xy, scores, range(91, 112)),
        "righthand_kpts": _flat_kpts(xy, scores, range(112, 133)),
    }


def export_predictions(
    images_root: str | Path,
    annotations_path: str | Path,
    model_path: str | Path,
    *, device: str = "cpu", max_persons: int | None = None,
    input_width: int = 288, input_height: int = 384,
) -> List[Dict[str, Any]]:
    images_root = Path(images_root).expanduser().resolve()
    annotations_path = Path(annotations_path).expanduser().resolve()
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in data.get("images", [])}
    anns = [x for x in data.get("annotations", []) if int(x.get("category_id", 1)) == 1]
    model = RTMWOpenCVDNN(model_path, input_width=input_width, input_height=input_height, device=device)
    image_cache: Dict[int, np.ndarray] = {}
    preds: List[Dict[str, Any]] = []
    for ann in anns:
        if max_persons is not None and len(preds) >= int(max_persons):
            break
        image_id = int(ann["image_id"])
        info = images.get(image_id)
        if info is None:
            continue
        if image_id not in image_cache:
            path = images_root / str(info["file_name"])
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            image_cache[image_id] = frame
        x, y, w, h = map(float, ann["bbox"])
        result = model.infer_one(image_cache[image_id], [x, y, x + w, y + h])
        preds.append(prediction_record(image_id, int(ann.get("id", len(preds))), result.keypoints_xy, result.scores))
    return preds


def official_xtcoco_eval(annotations_path: str | Path, predictions_path: str | Path) -> Dict[str, Any]:
    try:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    except Exception as exc:
        return {"status": "XTCOCOTOOLS_UNAVAILABLE", "error": repr(exc)}
    gt_file, pred_file = str(Path(annotations_path).resolve()), str(Path(predictions_path).resolve())
    specs = [
        ("body", "keypoints_body", np.asarray(SIGMAS_BODY)),
        ("foot", "keypoints_foot", np.asarray(SIGMAS_FOOT)),
        ("face", "keypoints_face", np.asarray(SIGMAS_FACE)),
        ("lefthand", "keypoints_lefthand", np.asarray(SIGMAS_HAND)),
        ("righthand", "keypoints_righthand", np.asarray(SIGMAS_HAND)),
        ("wholebody", "keypoints_wholebody", np.asarray(SIGMAS_WHOLE)),
    ]
    result: Dict[str, Any] = {"status": "COMPLETE", "metrics": {}}
    for label, iou_type, sigmas in specs:
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(pred_file)
        ev = COCOeval(coco_gt, coco_dt, iou_type, sigmas, use_area=True)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats = [float(x) for x in ev.stats]
        result["metrics"][label] = {
            "AP": stats[0], "AP50": stats[1], "AP75": stats[2],
            "AP_medium": stats[3], "AP_large": stats[4],
            "AR": stats[5] if len(stats) > 5 else None,
            "raw_stats": stats,
        }
    return result
