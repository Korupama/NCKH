#!/usr/bin/env python3
"""Build train-only RTMW-L pseudo labels for Phase 8A self-training.

Pseudo labels are kept separate from ``keypoints_133`` and are never accepted
as human ground truth by the Phase-8 validator. Validation/test tasks are
never pseudo-labeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage3_pose2d.crop_qa import analyze_crop
from stage3_pose2d.quality import evaluate_pose
from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN
from stage3_pose2d.schemas import Stage3Config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pseudo_labels(
    manifest: Path,
    model_path: Path,
    output: Path,
    *,
    min_median_score: float = 1.0,
    max_tasks: int | None = None,
) -> Dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    model = RTMWOpenCVDNN(model_path, input_width=288, input_height=384, device="cpu")
    config = Stage3Config()
    tasks = document.get("tasks") or []
    tasks_by_image: Dict[str, list[Dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_image.setdefault(str(task.get("image_path")), []).append(task)
    accepted = rejected = skipped = processed = 0

    for task in tasks:
        if task.get("split") != "train":
            if "pseudo_label_status" in task:
                raise ValueError(f"Pseudo-label already exists outside train split: {task['task_id']}")
            continue
        if max_tasks is not None and processed >= max_tasks:
            break
        image = cv2.imread(str(task["image_path"]))
        if image is None:
            task["pseudo_label_status"] = "SKIPPED_IMAGE_UNREADABLE"
            skipped += 1
            processed += 1
            continue
        result = model.infer_one(image, task["bbox_xyxy"])
        neighbors = [
            {"track_id": str(other.get("track_id") or other.get("task_id")), "bbox_xyxy": other["bbox_xyxy"]}
            for other in tasks_by_image.get(str(task.get("image_path")), [])
            if other is not task and other.get("bbox_xyxy")
        ]
        crop_diagnostics = analyze_crop(
            task["bbox_xyxy"],
            (image.shape[1], image.shape[0]),
            neighbors=neighbors,
            pose_xy=result.keypoints_xy,
            pose_scores=result.scores,
            bbox_padding=config.bbox_padding,
            input_size_wh=(config.rtmw_input_width, config.rtmw_input_height),
        )
        qa, records = evaluate_pose(
            result.keypoints_xy,
            result.scores,
            task["bbox_xyxy"],
            config,
            crop_diagnostics=crop_diagnostics,
        )
        median_score = float(qa.get("median_positive_raw_score", 0.0))
        crop_status = str(crop_diagnostics.get("crop_status", "UNKNOWN"))
        ownership_status = str(crop_diagnostics.get("ownership_status", "UNKNOWN"))
        accepted_by_gate = (
            qa.get("pose_status") == "VALID"
            and crop_status == "OK"
            and ownership_status in {"SUPPORTED", "UNKNOWN"}
            and median_score >= float(min_median_score)
        )
        task["pseudo_label_qa"] = qa
        task["pseudo_label_crop_diagnostics"] = crop_diagnostics
        task["pseudo_label_provenance"] = {
            "annotation_source": "PSEUDO_LABEL",
            "teacher_model": "RTMW-L",
            "teacher_model_path": str(model_path.resolve()),
            "teacher_model_sha256": _sha256(model_path),
            "input_size_wh": [config.rtmw_input_width, config.rtmw_input_height],
            "bbox_padding": config.bbox_padding,
            "is_ground_truth": False,
            "selection_policy": "Stage3_VALID_with_crop_ownership_QA_and_median_raw_score_floor",
            "min_median_raw_score": float(min_median_score),
        }
        if accepted_by_gate:
            task["pseudo_keypoints_133"] = [
                {
                    "index": int(i),
                    "x": record["x"],
                    "y": record["y"],
                    "raw_model_score": record["raw_model_score"],
                    "state": record["state"],
                }
                for i, record in enumerate(records)
            ]
            task["pseudo_label_status"] = "ACCEPTED"
            accepted += 1
        else:
            task.pop("pseudo_keypoints_133", None)
            task["pseudo_label_status"] = "REJECTED_QA"
            rejected += 1
        processed += 1

    document["schema_version"] = "stage3-phase8-pseudo-labels-1.0"
    document["status"] = "PSEUDO_LABEL_READY"
    document["train_only"] = True
    document["ground_truth_unchanged"] = True
    document["pseudo_label_summary"] = {
        "teacher_model": "RTMW-L",
        "teacher_model_sha256": _sha256(model_path),
        "processed_train_tasks": processed,
        "accepted_train_tasks": accepted,
        "rejected_train_tasks": rejected,
        "skipped_train_tasks": skipped,
        "min_median_raw_score": float(min_median_score),
        "validation_test_pseudo_labels_created": 0,
        "crop_ownership_qa_enabled": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return document["pseudo_label_summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rtmw-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-median-score", type=float, default=1.0)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build_pseudo_labels(args.manifest, args.rtmw_model, args.output, min_median_score=args.min_median_score, max_tasks=args.max_tasks), indent=2))


if __name__ == "__main__":
    main()
