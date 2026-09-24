import json
from pathlib import Path

from ball_localization.evaluation.contact_benchmark_v1 import benchmark_contact_manifest
from ball_localization.evaluation.protocol_v1 import compose_primary_protocol_report


def test_primary_protocol_maps_ble_x_to_ball_x():
    detector = {
        "status": "COMPLETE",
        "missing_images": 0,
        "evaluated_images": 100,
        "metrics": {
            "AP50": 0.8,
            "mAP50_95": 0.5,
            "precision": 0.9,
            "recall": 0.85,
            "CandidateRecall@5": 0.95,
            "CenterErrorPx_median": 2.0,
            "CenterErrorPx_P90": 5.0,
        },
    }
    oracle = {
        "status": "COMPLETE",
        "metrics": {
            "BLE_X_MAE_m": 0.10,
            "BLE_X_median_m": 0.08,
            "BLE_X_P90_m": 0.20,
            "BLE_X_P95_m": 0.25,
            "coverage": 1.0,
            "MAE_3D_m": 0.5,
        },
    }
    e2e = {
        "status": "COMPLETE",
        "metrics": {
            "BLE_X_MAE_m": 0.15,
            "BLE_X_median_m": 0.11,
            "BLE_X_P90_m": 0.30,
            "BLE_X_P95_m": 0.40,
            "coverage": 0.9,
            "MAE_3D_m": 0.7,
        },
    }
    report = compose_primary_protocol_report(
        detector_report=detector,
        oracle_report=oracle,
        e2e_report=e2e,
        split="test",
        smoke=False,
    )
    assert report["scientific_claim_ready"] is True
    assert report["stage6b_e2e_top1"]["mae_m"] == 0.15
    assert abs(report["error_budget"]["e2e_minus_oracle_ball_x_mae_m"] - 0.05) < 1e-9


def test_truncated_primary_is_smoke():
    base = {"status": "COMPLETE", "metrics": {}}
    detector = {**base, "missing_images": 0, "metrics": {"AP50": 0.1}}
    report = compose_primary_protocol_report(
        detector_report=detector,
        oracle_report=base,
        e2e_report=base,
        split="test",
        smoke=True,
    )
    assert report["status"] == "SMOKE"
    assert report["scientific_claim_ready"] is False


def test_contact_manifest_missing_assignment_counts_as_error(tmp_path: Path):
    good_state = {
        "selected_frame_ball": {
            "frame_index": 100,
            "status": "VALID_CONTACT_GROUND_PLANE",
            "X_world_m": 10.2,
            "contact": {"status": "SUPPORTED", "track_id": "track_006", "region": "FOOT"},
            "localization": {"X_world_m": 10.2},
        }
    }
    missing_assignment_state = {
        "selected_frame_ball": {
            "frame_index": 200,
            "status": "DEGRADED_CONTACT_FALLBACK",
            "X_world_m": 20.0,
            "contact": {"status": "AMBIGUOUS", "track_id": None, "region": "FOOT"},
            "localization": {"X_world_m": 20.0},
        }
    }
    (tmp_path / "a.json").write_text(json.dumps(good_state), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(missing_assignment_state), encoding="utf-8")
    manifest = {
        "metadata": {"dataset": "unit-test", "split": "test", "frozen": True},
        "cases": [
            {"case_id": "a", "state_json": "a.json", "gt_track_id": "track_006", "gt_region": "FOOT", "gt_x_m": 10.0},
            {"case_id": "b", "state_json": "b.json", "gt_track_id": "track_010", "gt_region": "FOOT", "gt_x_m": 20.0},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = benchmark_contact_manifest(manifest_path, tmp_path / "out")
    assert report["metrics"]["contact_track_accuracy"] == 0.5
    assert report["metrics"]["contact_assignment_coverage"] == 0.5
    # Region is only scored when Stage 6 actually assigned a toucher.
    assert report["metrics"]["contact_region_accuracy"] == 0.5
    assert abs(report["metrics"]["Ball_X_MAE_m"] - 0.1) < 1e-9


def test_geometry_diagnostic_report_is_not_production_metric():
    from ball_localization.evaluation.protocol_v1 import compose_geometry_diagnostic_report

    decomp = {
        "status": "COMPLETE",
        "variants_all_evaluable": {
            "GT_CENTER_GT_DIAMETER": {"mae_m": 1.0, "coverage": 1.0},
            "PRED_CENTER_GT_DIAMETER": {"mae_m": 1.2, "coverage": 0.9},
            "GT_CENTER_PRED_DIAMETER": {"mae_m": 2.0, "coverage": 0.9},
            "PRED_CENTER_PRED_DIAMETER": {"mae_m": 2.3, "coverage": 0.8},
        },
        "error_attribution_common_frames": {"diameter_only_delta_vs_oracle_m": 1.0},
        "variants_top1_iou50_matched": {},
        "counts": {"records": 10},
    }
    best = {
        "status": "COMPLETE",
        "metrics": {
            "BLE_X_MAE_m": 1.5,
            "BLE_X_median_m": 1.0,
            "BLE_X_P90_m": 3.0,
            "BLE_X_P95_m": 4.0,
            "coverage": 0.85,
        },
    }
    report = compose_geometry_diagnostic_report(
        decomposition_report=decomp,
        best_iou_report=best,
        split="test",
        smoke=False,
    )
    assert report["scientific_diagnostic_ready"] is True
    assert report["production_metric"] is False
    assert abs(report["best_iou_diagnostic"]["ball_x_mae_gain_vs_top1_m"] - 0.8) < 1e-9


def test_center_diameter_attribution_uses_common_frame_mae_deltas():
    from ball_localization.evaluation.benchmark_3d_decomposition_v12 import (
        _attribution,
        VARIANT_ORACLE,
        VARIANT_CENTER,
        VARIANT_DIAMETER,
        VARIANT_E2E,
    )

    common = {
        VARIANT_ORACLE: {"mae_m": 1.0},
        VARIANT_CENTER: {"mae_m": 1.2},
        VARIANT_DIAMETER: {"mae_m": 2.1},
        VARIANT_E2E: {"mae_m": 2.4},
    }
    out = _attribution(common)
    assert abs(out["center_only_delta_vs_oracle_m"] - 0.2) < 1e-9
    assert abs(out["diameter_only_delta_vs_oracle_m"] - 1.1) < 1e-9
    assert abs(out["combined_delta_vs_oracle_m"] - 1.4) < 1e-9
    assert abs(out["nonadditive_interaction_residual_m"] - 0.1) < 1e-9


def test_v12_translate_box_preserves_width_height_and_new_center():
    import numpy as np
    from ball_localization.evaluation.benchmark_3d_decomposition_v12 import _translate_box
    box = [10.0, 20.0, 18.0, 26.0]
    moved = _translate_box(box, np.asarray([100.0, 200.0]))
    assert abs((moved[2] - moved[0]) - 8.0) < 1e-12
    assert abs((moved[3] - moved[1]) - 6.0) < 1e-12
    assert abs((moved[0] + moved[2]) / 2.0 - 100.0) < 1e-12
    assert abs((moved[1] + moved[3]) / 2.0 - 200.0) < 1e-12


def test_geometry_consistency_checks_coverage_as_well_as_mae():
    from ball_localization.evaluation.protocol_v1 import compose_geometry_diagnostic_report
    decomp = {
        "status": "COMPLETE",
        "variants_all_evaluable": {
            "GT_CENTER_GT_DIAMETER": {"mae_m": 1.0, "coverage": 0.95},
            "PRED_CENTER_GT_DIAMETER": {"mae_m": 1.1, "coverage": 0.90},
            "GT_CENTER_PRED_DIAMETER": {"mae_m": 2.0, "coverage": 0.80},
            "PRED_CENTER_PRED_DIAMETER": {"mae_m": 2.3, "coverage": 0.75},
        },
        "error_attribution_common_frames": {},
        "variants_top1_iou50_matched": {},
        "counts": {"records": 10},
        "top1_observation_error": {},
        "e2e_by_absolute_diameter_relative_error": {},
    }
    best = {"status": "COMPLETE", "metrics": {"BLE_X_MAE_m": 2.0, "coverage": 0.75}}
    primary = {
        "stage6b_oracle_geometry": {"mae_m": 1.0, "coverage": 0.95},
        "stage6b_e2e_top1": {"mae_m": 2.3, "coverage": 0.75},
    }
    report = compose_geometry_diagnostic_report(
        decomposition_report=decomp,
        best_iou_report=best,
        split="test",
        smoke=False,
        primary_report=primary,
    )
    checks = report["primary_consistency_check"]
    assert checks["oracle_coverage"]["status"] == "MATCH"
    assert checks["top1_e2e_coverage"]["status"] == "MATCH"
