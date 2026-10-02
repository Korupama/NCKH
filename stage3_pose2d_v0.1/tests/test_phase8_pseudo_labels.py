import json

import numpy as np

from tools.build_pseudo_labels import build_pseudo_labels
from tools.export_pseudo_coco import export_pseudo_coco
from tools.prepare_annotation_tasks import _split_sequences
from tools.validate_annotations import validate


class _FakePoseResult:
    def __init__(self):
        self.keypoints_xy = np.tile(np.asarray([[16.0, 24.0]], dtype=np.float32), (133, 1))
        self.scores = np.full(133, 3.0, dtype=np.float32)


class _FakeTeacher:
    def __init__(self, *_args, **_kwargs):
        pass

    def infer_one(self, *_args, **_kwargs):
        return _FakePoseResult()


def _manifest(tmp_path):
    import cv2

    tasks = []
    for split, sequence in (("train", "seq_train"), ("validation", "seq_val"), ("test", "seq_test")):
        image = tmp_path / f"frame_{split}.jpg"
        cv2.imwrite(str(image), np.zeros((128, 128, 3), np.uint8))
        tasks.append({
            "task_id": f"{sequence}:1:1",
            "split": split,
            "sequence_id": sequence,
            "image_id": f"{sequence}:1",
            "image_path": str(image),
            "track_id": 1,
            "bbox_xyxy": [5.0, 9.0, 27.0, 39.0],
            "keypoints_133": [{"index": i, "x": None, "y": None, "visibility": None} for i in range(133)],
            "annotation_source": "UNANNOTATED",
            "review_status": "PENDING",
        })
    return {
        "pose_schema": {"name": "COCO_WHOLEBODY_133", "count": 133},
        "split_rule": {"sequence_splits": {"train": ["seq_train"], "validation": ["seq_val"], "test": ["seq_test"]}},
        "tasks": tasks,
    }


def test_sequence_split_is_disjoint_and_deterministic():
    split = _split_sequences([f"seq_{i}" for i in range(20)], 7)
    all_ids = [sid for values in split.values() for sid in values]
    assert len(all_ids) == len(set(all_ids)) == 20
    assert split == _split_sequences([f"seq_{i}" for i in range(20)], 7)


def test_pseudo_labels_are_train_only_and_do_not_change_ground_truth(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    output = tmp_path / "pseudo.json"
    monkeypatch.setattr("tools.build_pseudo_labels.RTMWOpenCVDNN", _FakeTeacher)
    monkeypatch.setattr(
        "tools.build_pseudo_labels.evaluate_pose",
        lambda xy, scores, bbox, config, **kwargs: (
            {
                "pose_status": "VALID",
                "median_positive_raw_score": 3.0,
            },
            [
                {
                    "index": i,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "raw_model_score": float(scores[i]),
                    "state": "VALID",
                }
                for i in range(133)
            ],
        ),
    )

    summary = build_pseudo_labels(manifest, model, output, min_median_score=1.0)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary["accepted_train_tasks"] == 1
    assert summary["validation_test_pseudo_labels_created"] == 0
    assert result["ground_truth_unchanged"] is True
    train, val, test = result["tasks"]
    assert train["pseudo_label_status"] == "ACCEPTED"
    assert len(train["pseudo_keypoints_133"]) == 133
    assert "pseudo_label_status" not in val and "pseudo_label_status" not in test
    assert all(point["x"] is None for point in train["keypoints_133"])
    assert validate(output, require_complete=False)["status"] == "PASS"


def test_pseudo_coco_export_is_train_only_and_non_ground_truth(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    pseudo = tmp_path / "pseudo.json"
    coco = tmp_path / "pseudo_coco.json"
    monkeypatch.setattr("tools.build_pseudo_labels.RTMWOpenCVDNN", _FakeTeacher)
    monkeypatch.setattr(
        "tools.build_pseudo_labels.evaluate_pose",
        lambda xy, scores, bbox, config, **kwargs: (
            {"pose_status": "VALID", "median_positive_raw_score": 3.0},
            [
                {
                    "index": i,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "raw_model_score": float(scores[i]),
                    "state": "VALID",
                }
                for i in range(133)
            ],
        ),
    )
    build_pseudo_labels(manifest, model, pseudo, min_median_score=1.0)

    export_pseudo_coco(pseudo, coco)
    result = json.loads(coco.read_text(encoding="utf-8"))
    assert result["annotation_source"] == "PSEUDO_LABEL"
    assert result["is_ground_truth"] is False
    assert result["training_only"] is True
    assert len(result["annotations"]) == 1
    assert len(result["annotations"][0]["keypoints"]) == 51
    assert len(result["annotations"][0]["foot_kpts"]) == 18
    assert len(result["annotations"][0]["face_kpts"]) == 204


def test_pseudo_label_rejects_high_overlap_even_with_strong_model_scores(monkeypatch, tmp_path):
    manifest_data = _manifest(tmp_path)
    train = manifest_data["tasks"][0]
    neighbor = dict(train)
    neighbor["task_id"] = "seq_train:1:2"
    neighbor["track_id"] = 2
    neighbor["bbox_xyxy"] = [5.0, 9.0, 27.0, 39.0]
    manifest_data["tasks"].append(neighbor)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    output = tmp_path / "pseudo.json"
    monkeypatch.setattr("tools.build_pseudo_labels.RTMWOpenCVDNN", _FakeTeacher)
    monkeypatch.setattr(
        "tools.build_pseudo_labels.evaluate_pose",
        lambda xy, scores, bbox, config, **kwargs: (
            {"pose_status": "VALID", "median_positive_raw_score": 3.0},
            [
                {
                    "index": i,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "raw_model_score": float(scores[i]),
                    "state": "VALID",
                }
                for i in range(133)
            ],
        ),
    )
    monkeypatch.setattr(
        "tools.build_pseudo_labels.analyze_crop",
        lambda *args, **kwargs: {"crop_status": "HIGH_OVERLAP", "ownership_status": "SUPPORTED"},
    )

    summary = build_pseudo_labels(manifest, model, output, min_median_score=1.0)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary["accepted_train_tasks"] == 0
    assert result["tasks"][0]["pseudo_label_status"] == "REJECTED_QA"
