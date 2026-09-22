import json

from benchmark.pose23 import evaluate_pose23


def _gt_point(index, visibility="VISIBLE"):
    if visibility in {"OUT_OF_FRAME", "NOT_ANNOTATED"}:
        x = y = None
    else:
        x, y = float(index * 2), float(index * 3)
    return {
        "index": index,
        "name": f"kp_{index}",
        "x": x,
        "y": y,
        "visibility": visibility,
    }


def _pred_point(index, *, x=None, y=None, kind=None, estimate=None):
    return {
        "index": index,
        "name": f"kp_{index}",
        "x": float(index * 2 if x is None else x),
        "y": float(index * 3 if y is None else y),
        "coordinate_evidence_kind": kind,
        "temporal_estimate_xy": estimate,
        "state": "VALID",
    }


def _fixture(tmp_path):
    gt_points = [_gt_point(i) for i in range(23)]
    gt_points[22] = _gt_point(22, "OCCLUDED")
    gt = {
        "annotations": [{
            "id": "a",
            "bbox_xyxy": [0, 0, 100, 200],
            "keypoints_23": gt_points,
            "tags": {
                "scale_bin": "SMALL", "occlusion_level": "PARTIAL",
                "motion_blur": "MILD", "view": "SIDE",
                "border_truncated": False, "crowd_level": "CROWDED",
            },
        }],
        "evaluation_policy": {"primary": ["VISIBLE"], "secondary": ["OCCLUDED"]},
    }
    predictions = [_pred_point(i, kind="RAW_OBSERVED") for i in range(23)]
    # 8 px / max bbox side 200 = 0.04: passes @.05 and fails @.01.
    predictions[0]["x"] += 8.0
    # This point has coordinates but is explicitly temporal-only. It must not
    # contribute to primary PCK or raw coverage.
    predictions[1]["coordinate_evidence_kind"] = "TEMPORAL_ESTIMATE_ONLY"
    predictions[1]["temporal_estimate_xy"] = [2.0, 3.0]
    predictions[22]["coordinate_evidence_kind"] = "RAW_OBSERVED"
    pred = {"predictions": [{"id": "a", "keypoints_23": predictions}]}
    gt_path, pred_path = tmp_path / "gt.json", tmp_path / "pred.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    pred_path.write_text(json.dumps(pred), encoding="utf-8")
    return gt_path, pred_path


def test_task_pose23_reports_keypoints_groups_slices_and_raw_provenance(tmp_path):
    gt_path, pred_path = _fixture(tmp_path)
    report = evaluate_pose23(gt_path, pred_path)
    metrics = report["metrics"]

    assert report["schema_version"] == "stage3-pose23-eval-1.1"
    assert metrics["PCK@0.05"] is not None
    assert metrics["per_keypoint"]["nose"]["PCK@0.05"] == 1.0
    assert metrics["per_keypoint"]["left_eye"]["raw_observed_count"] == 0
    assert metrics["groups"]["heels"]["gt_count"] == 1
    assert metrics["primary"]["overall"]["gt_count"] == 22
    assert metrics["secondary"]["overall"]["gt_count"] == 1
    assert metrics["provenance"]["temporal_estimate_only_points"] == 1
    assert metrics["provenance"]["primary_metric_coordinate_source"] == "RAW_OBSERVED_ONLY"
    assert metrics["difficulty_slices"]["scale_bin"]["SMALL"]["sample_count"] == 1


def test_task_pose23_missing_prediction_is_counted_not_dropped(tmp_path):
    gt_path, pred_path = _fixture(tmp_path)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    payload["predictions"][0]["keypoints_23"][2]["x"] = None
    payload["predictions"][0]["keypoints_23"][2]["y"] = None
    pred_path.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_pose23(gt_path, pred_path)
    nose = report["metrics"]["per_keypoint"]["right_eye"]
    assert nose["gt_count"] == 1
    assert nose["raw_observed_count"] == 0
    assert nose["PCK@0.05"] == 0.0


def test_temporal_estimate_is_diagnostic_only_and_oks_is_optional(tmp_path):
    gt_path, pred_path = _fixture(tmp_path)
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    point = payload["predictions"][0]["keypoints_23"][1]
    point["x"] = None
    point["y"] = None
    point["coordinate_evidence_kind"] = "TEMPORAL_ESTIMATE_ONLY"
    point["temporal_estimate_xy"] = [2.0, 3.0]
    pred_path.write_text(json.dumps(payload), encoding="utf-8")

    primary = evaluate_pose23(gt_path, pred_path)
    assert primary["metrics"]["per_keypoint"]["left_eye"]["raw_observed_count"] == 0
    assert "OKS" not in primary["metrics"]

    from benchmark.pose23_task import evaluate_pose23_task
    diagnostic = evaluate_pose23_task(gt_path, pred_path, score_temporal_estimates=True, include_oks=True)
    assert diagnostic["metrics"]["temporal_estimate_diagnostic"]["per_keypoint"]["left_eye"]["PCK@0.05"] == 1.0
    assert diagnostic["metrics"]["primary"]["overall"]["raw_observed_count"] == 21
    assert diagnostic["metrics"]["primary"]["overall"]["OKS"] is not None
