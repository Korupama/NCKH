#!/usr/bin/env python3
"""Prepare a shot-disjoint WholeBody133 annotation task manifest.

SoccerNet-GSR supplies images, person boxes, roles and track IDs. It does not
supply pose keypoints. This tool deliberately emits empty keypoint slots and
marks every task as ``PENDING``; RTMW predictions must never be silently
promoted to training ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


WHOLEBODY_COUNT = 133
PERSON_CATEGORY_IDS = {1, 2}  # player, goalkeeper


def _stable_order(sequence_ids: Sequence[str], seed: int) -> List[str]:
    return sorted(sequence_ids, key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())


def _split_sequences(sequence_ids: Sequence[str], seed: int) -> Dict[str, List[str]]:
    ordered = _stable_order(sequence_ids, seed)
    n = len(ordered)
    train_n = max(1, int(round(n * 0.70)))
    val_n = max(1, int(round(n * 0.15)))
    if train_n + val_n >= n:
        val_n = max(1, n - train_n - 1)
    return {
        "train": ordered[:train_n],
        "validation": ordered[train_n:train_n + val_n],
        "test": ordered[train_n + val_n:],
    }


def _evenly_spaced(items: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    if count <= 0 or len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[i] for i in indexes]


def _empty_keypoints() -> List[Dict[str, Any]]:
    return [
        {"index": i, "x": None, "y": None, "visibility": None}
        for i in range(WHOLEBODY_COUNT)
    ]


def prepare(root: Path, *, split: str, seed: int, frames_per_sequence: int, roles: set[str]) -> Dict[str, Any]:
    split_root = (root / split).resolve()
    sequence_dirs = sorted(p for p in split_root.iterdir() if p.is_dir())
    sequence_splits = _split_sequences([p.name for p in sequence_dirs], seed)
    split_by_sequence = {sid: name for name, ids in sequence_splits.items() for sid in ids}
    tasks: List[Dict[str, Any]] = []

    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name
        label_path = sequence_dir / "Labels-GameState.json"
        if not label_path.is_file():
            continue
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        images = {str(x["image_id"]): x for x in labels.get("images", [])}
        anns_by_image: Dict[str, List[Mapping[str, Any]]] = {}
        for ann in labels.get("annotations", []):
            attrs = ann.get("attributes") or {}
            role = str(attrs.get("role", "")).lower()
            if role not in roles or int(ann.get("category_id", -1)) not in PERSON_CATEGORY_IDS:
                continue
            anns_by_image.setdefault(str(ann["image_id"]), []).append(ann)
        eligible_images = [images[k] for k in sorted(anns_by_image) if k in images]
        selected_images = _evenly_spaced(eligible_images, frames_per_sequence)
        assignment = split_by_sequence[sequence_id]
        for image in selected_images:
            image_id = str(image["image_id"])
            image_path = sequence_dir / str(labels["info"].get("im_dir", "img1")) / str(image["file_name"])
            for ann in sorted(anns_by_image[image_id], key=lambda x: (int(x.get("track_id", -1)), str(x["id"]))):
                box = ann.get("bbox_image") or {}
                x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
                attrs = ann.get("attributes") or {}
                task_id = f"{sequence_id}:{image_id}:{ann.get('track_id', ann.get('id'))}"
                tasks.append({
                    "task_id": task_id,
                    "split": assignment,
                    "sequence_id": sequence_id,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "track_id": ann.get("track_id"),
                    "role": attrs.get("role"),
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "keypoints_133": _empty_keypoints(),
                    "annotation_source": "UNANNOTATED",
                    "review_status": "PENDING",
                    "preannotation_policy": "RTMW predictions may assist the annotator but are not ground truth",
                })

    return {
        "schema_version": "stage3-phase8-wholebody133-annotation-tasks-1.0",
        "status": "ANNOTATION_READY",
        "train_ready": False,
        "source_dataset": "SoccerNet-GSR",
        "source_split": split,
        "source_root": str(root.resolve()),
        "source_rights": "USER_AUTHORIZED_LOCAL_USE_REQUIRES_CONFIRMATION_BEFORE_TRAINING",
        "pose_schema": {"name": "COCO_WHOLEBODY_133", "count": WHOLEBODY_COUNT},
        "split_rule": {
            "unit": "sequence_id",
            "seed": seed,
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "sequence_splits": sequence_splits,
        },
        "sampling": {"frames_per_sequence": frames_per_sequence, "roles": sorted(roles)},
        "annotation_requirements": {
            "human_verified": True,
            "all_133_points_required": True,
            "reviewer_id_required": True,
            "review_timestamp_required": True,
            "predictions_are_not_ground_truth": True,
        },
        "counts": {
            "sequences": len(sequence_splits["train"] + sequence_splits["validation"] + sequence_splits["test"]),
            "tasks": len(tasks),
            "tasks_by_split": {name: sum(t["split"] == name for t in tasks) for name in sequence_splits},
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsr-root", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--frames-per-sequence", type=int, default=5)
    parser.add_argument("--roles", default="player,goalkeeper")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roles = {x.strip().lower() for x in args.roles.split(",") if x.strip()}
    result = prepare(args.gsr_root, split=args.split, seed=args.seed, frames_per_sequence=args.frames_per_sequence, roles=roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "counts": result["counts"], "splits": result["split_rule"]["sequence_splits"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
