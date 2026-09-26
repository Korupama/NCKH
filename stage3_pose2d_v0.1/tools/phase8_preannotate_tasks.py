#!/usr/bin/env python3
"""Add RTMW-L pre-annotations to Phase-8 tasks without changing GT fields.

The generated ``model_preannotation_133`` field is explicitly separate from
``keypoints_133``. The validator never accepts model pre-annotations as
training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import cv2

from stage3_pose2d.rtmw_onnx import RTMWOpenCVDNN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preannotate(manifest: Path, model_path: Path, output: Path, max_tasks: int | None = None) -> Dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    model = RTMWOpenCVDNN(model_path, input_width=288, input_height=384, device="cpu")
    tasks = document.get("tasks") or []
    processed = 0
    skipped = 0
    for task in tasks:
        if max_tasks is not None and processed >= max_tasks:
            break
        image = cv2.imread(str(task["image_path"]))
        if image is None:
            skipped += 1
            continue
        result = model.infer_one(image, task["bbox_xyxy"])
        task["model_preannotation_133"] = [
            {
                "index": int(i),
                "x": float(x),
                "y": float(y),
                "raw_model_score": float(score),
            }
            for i, ((x, y), score) in enumerate(zip(result.keypoints_xy, result.scores))
        ]
        task["model_preannotation_provenance"] = {
            "model": "RTMW-L",
            "model_path": str(model_path.resolve()),
            "model_sha256": _sha256(model_path),
            "is_ground_truth": False,
            "review_required": True,
        }
        processed += 1
    document["preannotation"] = {
        "status": "MODEL_PREANNOTATION_ONLY",
        "processed_tasks": processed,
        "skipped_tasks": skipped,
        "remaining_tasks": max(0, len(tasks) - processed),
        "ground_truth_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return document["preannotation"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rtmw-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(preannotate(args.manifest, args.rtmw_model, args.output, args.max_tasks), indent=2))


if __name__ == "__main__":
    main()
