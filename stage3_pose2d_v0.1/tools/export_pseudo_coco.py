#!/usr/bin/env python3
"""Export accepted train-only pseudo labels to COCO-WholeBody JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def export_pseudo_coco(manifest: Path, output: Path) -> Dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    images: Dict[str, Dict[str, Any]] = {}
    annotations = []
    next_annotation_id = 1
    for task in document.get("tasks") or []:
        if task.get("split") != "train" or task.get("pseudo_label_status") != "ACCEPTED":
            continue
        points = task.get("pseudo_keypoints_133") or []
        if len(points) != 133:
            raise ValueError(f"Accepted task lacks 133 pseudo keypoints: {task.get('task_id')}")
        image_id = f"{task['sequence_id']}:{task['image_id']}"
        if image_id not in images:
            image = Path(task["image_path"])
            images[image_id] = {
                "id": image_id,
                "file_name": str(image),
                "width": None,
                "height": None,
            }
        x1, y1, x2, y2 = map(float, task["bbox_xyxy"])
        groups = {"body": [], "foot": [], "face": [], "left_hand": [], "right_hand": []}
        visible = 0
        for index, point in enumerate(points):
            x, y = point.get("x"), point.get("y")
            score = float(point.get("raw_model_score") or 0.0)
            if x is None or y is None:
                encoded = [0.0, 0.0, 0]
            else:
                # COCO visibility is a training hint here, never a claim of
                # human annotation. The pseudo provenance remains in attrs.
                visibility = 2 if score > 0.0 else 0
                encoded = [float(x), float(y), visibility]
                visible += visibility > 0
            if index < 17:
                groups["body"].extend(encoded)
            elif index < 23:
                groups["foot"].extend(encoded)
            elif index < 91:
                groups["face"].extend(encoded)
            elif index < 112:
                groups["left_hand"].extend(encoded)
            else:
                groups["right_hand"].extend(encoded)
        annotations.append({
            "id": next_annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "area": max(0.0, (x2 - x1) * (y2 - y1)),
            "iscrowd": 0,
            "num_keypoints": int(visible),
            "keypoints": groups["body"],
            "foot_kpts": groups["foot"],
            "face_kpts": groups["face"],
            "lefthand_kpts": groups["left_hand"],
            "righthand_kpts": groups["right_hand"],
            "segmentation": [],
            "pseudo_label": True,
            "pseudo_label_task_id": task["task_id"],
        })
        next_annotation_id += 1
    result = {
        "schema_version": "stage3-phase8-pseudo-coco-wholebody-1.0",
        "annotation_source": "PSEUDO_LABEL",
        "is_ground_truth": False,
        "training_only": True,
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": list(WHOLEBODY_KEYPOINT_NAMES),
            "skeleton": [],
        }],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"images": len(result["images"]), "annotations": len(annotations), "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_pseudo_coco(args.manifest, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
