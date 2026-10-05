#!/usr/bin/env python3
"""Validate human-reviewed WholeBody133 fine-tuning annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def validate(path: Path, require_complete: bool = True) -> Dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    errors: List[str] = []
    if doc.get("pose_schema", {}).get("name") != "COCO_WHOLEBODY_133":
        errors.append("pose_schema must be COCO_WHOLEBODY_133")
    if int(doc.get("pose_schema", {}).get("count", 0)) != 133:
        errors.append("pose_schema.count must be 133")
    tasks = doc.get("tasks") or []
    if doc.get("preannotation", {}).get("status") == "MODEL_PREANNOTATION_ONLY":
        # This is allowed as an annotation-workflow artifact, but the model
        # field is never inspected as ground truth below.
        pass
    seen = set()
    sequence_split = {}
    for split, ids in (doc.get("split_rule", {}).get("sequence_splits") or {}).items():
        for sid in ids:
            if sid in sequence_split:
                errors.append(f"sequence appears in multiple splits: {sid}")
            sequence_split[sid] = split
    for task in tasks:
        key = (str(task.get("sequence_id")), str(task.get("image_id")), str(task.get("track_id")))
        if key in seen:
            errors.append(f"duplicate task: {task.get('task_id')}")
        seen.add(key)
        if sequence_split.get(str(task.get("sequence_id"))) != task.get("split"):
            errors.append(f"split mismatch: {task.get('task_id')}")
        points = task.get("keypoints_133") or []
        if require_complete:
            if len(points) != 133:
                errors.append(f"{task.get('task_id')}: expected 133 keypoints")
            if task.get("annotation_source") != "HUMAN_VERIFIED":
                errors.append(f"{task.get('task_id')}: annotation_source is not HUMAN_VERIFIED")
            if task.get("review_status") != "APPROVED":
                errors.append(f"{task.get('task_id')}: review_status is not APPROVED")
            if not task.get("reviewer_id") or not task.get("reviewed_at"):
                errors.append(f"{task.get('task_id')}: reviewer_id/reviewed_at required")
            for point in points:
                if point.get("x") is None or point.get("y") is None or point.get("visibility") not in (0, 1, 2):
                    errors.append(f"{task.get('task_id')}: incomplete keypoint record")
                    break
    return {"schema_version": "stage3-phase8-annotation-validation-1.0", "status": "PASS" if not errors else "FAIL", "tasks": len(tasks), "errors": errors[:100]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.annotations, require_complete=not args.allow_pending)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
