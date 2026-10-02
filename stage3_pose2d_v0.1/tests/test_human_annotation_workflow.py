from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from stage3_pose2d.wholebody133 import WHOLEBODY_KEYPOINT_NAMES
from tools.annotation_reviewer import ReviewStore
from tools.export_human_coco import export_human_coco
from tools.final_acceptance import (
    CHECKPOINT_SCHEMA,
    _checkpoint_integrity_report,
    _preannotation_report,
    _sequence_split_report,
)


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    image_path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))
    task = {
        "task_id": "seq_train:1:1",
        "split": "train",
        "sequence_id": "seq_train",
        "image_id": "1",
        "image_path": str(image_path),
        "track_id": 1,
        "role": "player",
        "bbox_xyxy": [10.0, 20.0, 50.0, 100.0],
        "keypoints_133": [
            {"index": i, "x": None, "y": None, "visibility": None}
            for i in range(133)
        ],
        "annotation_source": "UNANNOTATED",
        "review_status": "PENDING",
    }
    document = {
        "schema_version": "stage3-phase8-wholebody133-annotation-tasks-1.0",
        "status": "ANNOTATION_READY",
        "train_ready": False,
        "source_dataset": "SyntheticTest",
        "pose_schema": {"name": "COCO_WHOLEBODY_133", "count": 133},
        "split_rule": {"sequence_splits": {"train": ["seq_train"], "validation": [], "test": []}},
        "tasks": [task],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def _complete_points() -> list[dict]:
    return [
        {"index": i, "name": name, "x": float(20 + i), "y": float(30 + i), "visibility": 2}
        for i, name in enumerate(WHOLEBODY_KEYPOINT_NAMES)
    ]


def test_reviewer_writes_separate_manifest_and_blocks_incomplete_approval(tmp_path):
    source, original = _manifest(tmp_path)
    output = tmp_path / "reviewed.json"
    store = ReviewStore(source, output, None)
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="all 133"):
        store.save({
            "task_id": "seq_train:1:1",
            "keypoints_133": [],
            "review_status": "APPROVED",
            "reviewer_id": "reviewer-1",
        })

    saved = store.save({
        "task_id": "seq_train:1:1",
        "keypoints_133": _complete_points(),
        "review_status": "APPROVED",
        "reviewer_id": "reviewer-1",
    })
    assert saved["complete"] is True
    assert source.read_bytes() == original_bytes
    reviewed = json.loads(output.read_text(encoding="utf-8"))
    task = reviewed["tasks"][0]
    assert task["annotation_source"] == "HUMAN_VERIFIED"
    assert task["review_status"] == "APPROVED"
    assert task["reviewer_id"] == "reviewer-1"
    assert original["tasks"][0]["keypoints_133"][0]["x"] is None


def test_human_export_rejects_pending_and_exports_approved_only(tmp_path):
    source, _ = _manifest(tmp_path)
    output = tmp_path / "reviewed.json"
    store = ReviewStore(source, output, None)
    with pytest.raises(ValueError, match="HUMAN_VERIFIED"):
        export_human_coco(output, tmp_path / "pending.json")

    store.save({
        "task_id": "seq_train:1:1",
        "keypoints_133": _complete_points(),
        "review_status": "APPROVED",
        "reviewer_id": "reviewer-1",
    })
    result = export_human_coco(output, tmp_path / "human.json")
    exported = json.loads((tmp_path / "human.json").read_text(encoding="utf-8"))
    assert result["annotations"] == 1
    assert exported["annotation_source"] == "HUMAN_VERIFIED"
    assert exported["is_ground_truth"] is True
    assert exported["training_only"] is True
    assert len(exported["annotations"][0]["keypoints"]) == 51
    assert len(exported["annotations"][0]["foot_kpts"]) == 18
    assert len(exported["annotations"][0]["face_kpts"]) == 204


def test_reviewer_keeps_model_preannotation_separate(tmp_path):
    source, _ = _manifest(tmp_path)
    preannotation = tmp_path / "pre.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    task = copy.deepcopy(document["tasks"][0])
    task["model_preannotation_133"] = [{"index": i, "x": 1.0, "y": 2.0, "raw_model_score": 3.0} for i in range(133)]
    preannotation.write_text(json.dumps({"tasks": [task]}), encoding="utf-8")
    output = tmp_path / "reviewed.json"
    store = ReviewStore(source, output, preannotation)
    reviewed = store.task("seq_train:1:1")
    assert reviewed is not None
    assert reviewed["keypoints_133"][0]["x"] is None
    assert reviewed["model_preannotation_133"][0]["x"] == 1.0


def test_acceptance_requires_annotation_split_to_match_frozen_manifest():
    document = {
        "split_rule": {
            "sequence_splits": {
                "train": ["a"], "validation": ["b"], "test": ["c"],
            },
        },
        "tasks": [],
    }
    frozen = {
        "sequence_splits": {
            "train": ["a"], "validation": ["changed"], "test": ["c"],
        },
    }
    report = _sequence_split_report(document, frozen)
    assert report["matches_frozen_split_manifest"] is False
    assert report["passed"] is False


def test_acceptance_detects_unexpected_split_key():
    document = {
        "split_rule": {"sequence_splits": {"train": ["a"], "extra": ["z"]}},
        "tasks": [],
    }
    frozen = {"sequence_splits": {"train": ["a"]}}
    report = _sequence_split_report(document, frozen)
    assert report["matches_frozen_split_manifest"] is False
    assert report["frozen_split_mismatches"]["extra"]["unexpected_in_annotation_manifest"] == ["z"]


def test_preannotation_acceptance_checks_provenance_and_ground_truth_immutability():
    source_task = {
        "task_id": "seq:1:1",
        "keypoints_133": [],
        "annotation_source": "UNANNOTATED",
        "review_status": "PENDING",
        "reviewer_id": None,
        "reviewed_at": None,
    }
    model_sha = "a" * 64
    output_task = copy.deepcopy(source_task)
    output_task.update({
        "model_preannotation_133": [{"index": index} for index in range(133)],
        "model_preannotation_qa": {},
        "model_preannotation_hard_negative_flags": {},
        "model_preannotation_provenance": {
            "model_sha256": model_sha,
            "is_ground_truth": False,
            "review_required": True,
        },
    })
    source = {"schema_version": "v1", "tasks": [source_task]}
    preannotated = {
        "schema_version": "v1",
        "tasks": [output_task],
        "preannotation": {
            "status": "MODEL_PREANNOTATION_ONLY",
            "remaining_tasks": 0,
            "ground_truth_unchanged": True,
            "pseudo_labels_are_not_training_ground_truth": True,
        },
    }
    validation = {"status": "PASS", "tasks": 1}

    assert _preannotation_report(source, preannotated, validation, model_sha256=model_sha)["passed"]
    preannotated["tasks"][0]["review_status"] = "APPROVED"
    report = _preannotation_report(source, preannotated, validation, model_sha256=model_sha)
    assert report["passed"] is False
    assert report["ground_truth_and_review_fields_unchanged"] is False


def test_checkpoint_acceptance_validates_chunk_hash_and_task_coverage(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    chunks_dir = checkpoint / "chunks"
    chunks_dir.mkdir(parents=True)
    task_ids = ["seq:1:1"]
    manifest_sha = "b" * 64
    model_sha = "c" * 64
    config = {"bbox_padding": 1.25}
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    chunk_name = "chunk-000001.json"
    chunk_path = chunks_dir / chunk_name
    chunk_path.write_text(json.dumps({
        "schema_version": CHECKPOINT_SCHEMA,
        "updates": {task_ids[0]: {"model_preannotation_133": []}},
    }), encoding="utf-8")
    chunk_sha = hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    task_order_sha = hashlib.sha256(json.dumps(task_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    (checkpoint / "meta.json").write_text(json.dumps({
        "schema_version": CHECKPOINT_SCHEMA,
        "manifest_sha256": manifest_sha,
        "model_sha256": model_sha,
        "task_count": 1,
        "task_order_sha256": task_order_sha,
        "task_ground_truth_sha256": {task_ids[0]: "d" * 64},
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "chunk_files": [chunk_name],
        "chunk_sha256": {chunk_name: chunk_sha},
        "completed_task_ids": task_ids,
    }), encoding="utf-8")

    report = _checkpoint_integrity_report(
        checkpoint,
        task_ids=task_ids,
        manifest_sha256=manifest_sha,
        model_sha256=model_sha,
    )
    assert report["passed"] is True
    chunk_path.write_text(chunk_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    report = _checkpoint_integrity_report(
        checkpoint,
        task_ids=task_ids,
        manifest_sha256=manifest_sha,
        model_sha256=model_sha,
    )
    assert report["passed"] is False
    assert any(error.startswith("chunk_hash_mismatch:") for error in report["errors"])
