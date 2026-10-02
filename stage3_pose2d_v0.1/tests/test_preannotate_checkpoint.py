import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.preannotate_tasks import preannotate


class _FakeResult:
    def __init__(self, offset: float):
        self.keypoints_xy = np.tile(np.asarray([[16.0 + offset, 24.0]], dtype=np.float32), (133, 1))
        self.scores = np.full(133, 3.0, dtype=np.float32)


class _FakeModel:
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def infer_one(self, _image, bbox):
        offset = float(len(self.calls))
        self.calls.append(list(bbox))
        return _FakeResult(offset)


def _write_manifest(tmp_path: Path) -> Path:
    tasks = []
    for index in range(3):
        image_path = tmp_path / f"frame_{index}.jpg"
        assert cv2.imwrite(str(image_path), np.zeros((96, 128, 3), dtype=np.uint8))
        tasks.append({
            "task_id": f"seq:1:{index}",
            "split": "train",
            "sequence_id": "seq",
            "image_id": str(index),
            "image_path": str(image_path),
            "track_id": index,
            "bbox_xyxy": [10.0, 10.0, 50.0, 80.0],
            "keypoints_133": [
                {"index": point, "x": None, "y": None, "visibility": None}
                for point in range(133)
            ],
            "annotation_source": "UNANNOTATED",
            "review_status": "PENDING",
        })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": "stage3-phase8-wholebody133-annotation-tasks-1.0",
        "pose_schema": {"name": "COCO_WHOLEBODY_133", "count": 133},
        "tasks": tasks,
    }), encoding="utf-8")
    return path


def _patch_dependencies(monkeypatch):
    monkeypatch.setattr("tools.preannotate_tasks.RTMWOpenCVDNN", _FakeModel)
    monkeypatch.setattr(
        "tools.preannotate_tasks.analyze_crop",
        lambda *args, **kwargs: {
            "crop_status": "OK",
            "ownership_status": "SUPPORTED",
            "source_bbox_xyxy": [10.0, 10.0, 50.0, 80.0],
            "neighbor_max_crop_iou": 0.0,
        },
    )
    monkeypatch.setattr(
        "tools.preannotate_tasks.evaluate_pose",
        lambda *args, **kwargs: ({
            "pose_status": "VALID",
            "low_model_evidence_fraction": 0.0,
            "feet_completeness": 1.0,
            "geometry_valid": True,
        }, []),
    )


def test_preannotation_resume_skips_completed_tasks_and_preserves_gt(monkeypatch, tmp_path):
    _patch_dependencies(monkeypatch)
    _FakeModel.calls = []
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    output = tmp_path / "preannotated.json"
    checkpoint = tmp_path / "checkpoint"
    source_before = manifest.read_bytes()

    first = preannotate(manifest, model, output, max_tasks=1, checkpoint=checkpoint, chunk_size=1)
    assert first["completed_this_run"] == 1
    assert first["processed_tasks"] == 1
    assert first["remaining_tasks"] == 2
    assert len(_FakeModel.calls) == 1

    second = preannotate(manifest, model, output, max_tasks=1, checkpoint=checkpoint, resume=True, chunk_size=1)
    assert second["completed_this_run"] == 1
    assert second["processed_tasks"] == 2
    assert second["remaining_tasks"] == 1
    assert len(_FakeModel.calls) == 2
    assert manifest.read_bytes() == source_before

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["preannotation"]["ground_truth_unchanged"] is True
    assert all(point["x"] is None for point in result["tasks"][0]["keypoints_133"])
    assert all("keypoints_133" not in update for update in [
        json.loads(path.read_text(encoding="utf-8"))["updates"][task_id]
        for path in sorted((checkpoint / "chunks").glob("*.json"))
        for task_id in json.loads(path.read_text(encoding="utf-8"))["updates"]
    ])


def test_preannotation_resume_rejects_changed_manifest_provenance(monkeypatch, tmp_path):
    _patch_dependencies(monkeypatch)
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    output = tmp_path / "preannotated.json"
    checkpoint = tmp_path / "checkpoint"
    preannotate(manifest, model, output, max_tasks=1, checkpoint=checkpoint, chunk_size=1)

    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["tasks"][0]["bbox_xyxy"][0] += 1.0
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch|ground-truth fields changed"):
        preannotate(manifest, model, output, checkpoint=checkpoint, resume=True)


def test_preannotation_resume_rejects_tampered_chunk(monkeypatch, tmp_path):
    _patch_dependencies(monkeypatch)
    manifest = _write_manifest(tmp_path)
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"teacher")
    output = tmp_path / "preannotated.json"
    checkpoint = tmp_path / "checkpoint"
    preannotate(manifest, model, output, max_tasks=1, checkpoint=checkpoint, chunk_size=1)

    chunk = next((checkpoint / "chunks").glob("*.json"))
    chunk.write_text(chunk.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk hash mismatch"):
        preannotate(manifest, model, output, checkpoint=checkpoint, resume=True)
