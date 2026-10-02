#!/usr/bin/env python3
"""Export human-approved Stage-3 annotations to COCO-WholeBody JSON.

This exporter is deliberately stricter than the annotation-task validator:
it only exports one requested split and refuses pending, pseudo or incomplete
tasks. The resulting file is suitable as a training input manifest, but does
not train or promote a model by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _approved_points(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if task.get("annotation_source") != "HUMAN_VERIFIED":
        raise ValueError(f"{task.get('task_id')}: annotation_source is not HUMAN_VERIFIED")
    if task.get("review_status") != "APPROVED":
        raise ValueError(f"{task.get('task_id')}: review_status is not APPROVED")
    if not task.get("reviewer_id") or not task.get("reviewed_at"):
        raise ValueError(f"{task.get('task_id')}: reviewer_id/reviewed_at required")
    points = task.get("keypoints_133") or []
    if len(points) != 133:
        raise ValueError(f"{task.get('task_id')}: expected 133 human keypoints")
    for index, point in enumerate(points):
        if int(point.get("index", -1)) != index:
            raise ValueError(f"{task.get('task_id')}: keypoint index mismatch at {index}")
        if str(point.get("name", WHOLEBODY_KEYPOINT_NAMES[index])) != WHOLEBODY_KEYPOINT_NAMES[index]:
            raise ValueError(f"{task.get('task_id')}: keypoint name mismatch at {index}")
        if not _finite(point.get("x")) or not _finite(point.get("y")):
            raise ValueError(f"{task.get('task_id')}: incomplete coordinate at {index}")
        if point.get("visibility") not in (0, 1, 2):
            raise ValueError(f"{task.get('task_id')}: invalid visibility at {index}")
    return points


def _encode_points(points: Iterable[Mapping[str, Any]]) -> tuple[Dict[str, list[float]], int]:
    groups: Dict[str, list[float]] = {"body": [], "foot": [], "face": [], "left_hand": [], "right_hand": []}
    visible = 0
    for index, point in enumerate(points):
        encoded = [float(point["x"]), float(point["y"]), int(point["visibility"])]
        if int(point["visibility"]) > 0:
            visible += 1
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
    return groups, visible


def export_human_coco(manifest: Path, output: Path, *, split: str = "train") -> Dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("pose_schema", {}).get("name") != "COCO_WHOLEBODY_133":
        raise ValueError("manifest must use COCO_WHOLEBODY_133")
    tasks = [task for task in document.get("tasks") or [] if task.get("split") == split]
    if not tasks:
        raise ValueError(f"manifest contains no tasks for split {split!r}")

    images: Dict[str, Dict[str, Any]] = {}
    annotations = []
    sequences = set()
    for annotation_id, task in enumerate(tasks, 1):
        points = _approved_points(task)
        image_path = Path(str(task["image_path"])).expanduser().resolve()
        if not image_path.is_file():
            raise ValueError(f"{task.get('task_id')}: image does not exist: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"{task.get('task_id')}: image is unreadable: {image_path}")
        height, width = image.shape[:2]
        image_id = f"{task['sequence_id']}:{task['image_id']}"
        sequences.add(str(task["sequence_id"]))
        images.setdefault(image_id, {"id": image_id, "file_name": str(image_path), "width": int(width), "height": int(height)})
        x1, y1, x2, y2 = map(float, task["bbox_xyxy"])
        groups, visible = _encode_points(points)
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "area": max(0.0, (x2 - x1) * (y2 - y1)),
            "iscrowd": 0,
            "num_keypoints": visible,
            "keypoints": groups["body"],
            "foot_kpts": groups["foot"],
            "face_kpts": groups["face"],
            "lefthand_kpts": groups["left_hand"],
            "righthand_kpts": groups["right_hand"],
            "segmentation": [],
            "annotation_source": "HUMAN_VERIFIED",
            "reviewer_id": task["reviewer_id"],
            "reviewed_at": task["reviewed_at"],
            "task_id": task["task_id"],
        })

    result = {
        "schema_version": "stage3-phase7-human-coco-wholebody-1.0",
        "annotation_source": "HUMAN_VERIFIED",
        "is_ground_truth": True,
        "training_only": split == "train",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": _sha256(manifest),
        "source_dataset": document.get("source_dataset"),
        "split": split,
        "sequence_ids": sorted(sequences),
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
    return {"split": split, "images": len(images), "annotations": len(annotations), "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export human-approved COCO-WholeBody annotations")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_human_coco(args.manifest, args.output, split=args.split), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
