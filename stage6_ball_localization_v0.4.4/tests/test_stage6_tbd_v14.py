from pathlib import Path
import csv
import json

from ball_localization.evaluation.contact_production_v14 import benchmark_contact_production_manifest
from ball_localization.evaluation.footpass_manifest_v14 import build_footpass_contact_manifest
from ball_localization.evaluation.tbd_status_v14 import build_tbd_status_report


def _state(path: Path, *, track, region, x, fallback=False):
    payload = {
        "selected_frame_ball": {
            "frame_index": 100,
            "X_world_m": x,
            "center_xyz_world_m": [x, 0.0, 0.11],
            "method": "CONTACT_GROUND_PLANE" if not fallback else "MONOCULAR_BALL_SIZE_PRIOR_FALLBACK",
            "contact": {"track_id": track, "region": region, "status": "SUPPORTED" if track else "AMBIGUOUS"},
            "localization": {"X_world_m": x, "usable_for_offside": True},
            "fallback": {"used": fallback, "reason": None},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_contact_production_metrics(tmp_path):
    s1, s2 = tmp_path / "s1.json", tmp_path / "s2.json"
    _state(s1, track="track_1", region="FOOT", x=10.1)
    _state(s2, track=None, region="HEAD", x=20.4, fallback=True)
    manifest = {
        "metadata": {"dataset": "synthetic", "split": "test", "frozen": True},
        "cases": [
            {"case_id": "a", "state_json": str(s1), "gt_track_id": "track_1", "gt_region": "FOOT", "gt_x_m": 10.0},
            {"case_id": "b", "state_json": str(s2), "gt_track_id": "track_2", "gt_region": "HEAD", "gt_x_m": 20.0},
        ],
    }
    mp = tmp_path / "manifest.json"; mp.write_text(json.dumps(manifest), encoding="utf-8")
    report = benchmark_contact_production_manifest(mp, tmp_path / "out")
    c = report["metrics"]["contact_primary"]
    assert c["assignment_coverage"] == 0.5
    assert c["assigned_precision"] == 1.0
    assert c["overall_accuracy"] == 0.5
    b = report["metrics"]["foot_vs_nonfoot"]
    assert b["prediction_coverage"] == 0.5
    assert b["foot_precision"] == 1.0
    assert b["foot_recall"] == 1.0
    x = report["metrics"]["production_ball_x"]
    assert x["coverage"] == 1.0
    assert abs(x["mae_m"] - 0.25) < 1e-9
    assert report["metric_readiness"] == {"contact_track_or_actor": True, "foot_nonfoot": True, "production_ball_x": True}


def test_footpass_builder_validates_and_derives_header(tmp_path):
    play = {"keys": ["game_18_H1"], "events": {"game_18_H1": [[123, 0, 9, 5, 1, 0]]}}
    pp = tmp_path / "play.json"; pp.write_text(json.dumps(play), encoding="utf-8")
    bridge = tmp_path / "bridge.csv"
    with bridge.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["game_key", "frame", "team", "jersey", "action_class", "state_json"])
        w.writeheader(); w.writerow({"game_key": "game_18_H1", "frame": 123, "team": 0, "jersey": 9, "action_class": "Header", "state_json": "state.json"})
    out = tmp_path / "manifest.json"
    result = build_footpass_contact_manifest(playbyplay_json=pp, bridge_csv=bridge, output_json=out, frozen=True)
    assert result["cases"] == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["cases"][0]["gt_region"] == "HEAD"
    assert payload["cases"][0]["gt_region_source"] == "FOOTPASS_ACTION_SEMANTICS"


def test_tbd_status_ground_and_contact(tmp_path):
    ground = {
        "scientific_diagnostic_ready": True,
        "primary_near_ground_proxy": {
            "common_valid_frames": {
                "records": 100,
                "metrics": {
                    "GROUND_PLANE_PRED_CENTER": {"mae_m": 0.4, "p95_m": 0.8, "coverage": 0.96},
                    "SIZE_PRIOR_TOP1": {"mae_m": 1.2, "p95_m": 3.0, "coverage": 0.8},
                },
            }
        },
    }
    gp = tmp_path / "ground.json"; gp.write_text(json.dumps(ground), encoding="utf-8")
    contact = {
        "scientific_claim_ready": True,
        "metric_readiness": {"contact_track_or_actor": True, "foot_nonfoot": True, "production_ball_x": False},
        "metrics": {
            "contact_primary": {"assignment_coverage": 0.9, "assigned_precision": 0.96, "overall_accuracy": 0.864},
            "foot_vs_nonfoot": {"accuracy": 0.95, "foot_precision": 0.96, "foot_recall": 0.95},
            "production_ball_x": {"mae_m": None, "p95_m": None, "coverage": None},
        },
        "project_target_assessment": {
            "contact_research_target": True, "contact_offside_target": True,
            "foot_nonfoot_research_target": True, "foot_nonfoot_offside_target": True,
            "ball_x_strong_research_target": None, "ball_x_offside_target": None, "ball_x_stretch_target": None,
        },
    }
    cp = tmp_path / "contact.json"; cp.write_text(json.dumps(contact), encoding="utf-8")
    report = build_tbd_status_report(output_dir=tmp_path / "status", ground_report=gp, contact_report=cp)
    assert report["ground_plane_ball_x"]["feasibility_50pct_improvement"] == "PASS"
    assert report["contact_and_production"]["contact"]["offside_target"] == "PASS"
    assert "temporal_hybrid_ball_x" in report["remaining_tbd"]
