import json
from pathlib import Path

from stage7_game_state.benchmark_v2 import (
    evaluate_manifest_v2,
    manifest_template,
    reference_report,
)


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _case_files(tmp_path: Path, *, ref_role="REFEREE"):
    s1 = _write(tmp_path / "s1.json", {"view": {"centre_ray_pitch_hit_m": [20.0, 0.0, 0.0]}})
    s5 = _write(tmp_path / "s5.json", {
        "players": [
            {"track_id": "a", "team_id": 0, "role": "PLAYER"},
            {"track_id": "b", "team_id": 0, "role": "GOALKEEPER"},
            {"track_id": "c", "team_id": 1, "role": "PLAYER"},
            {"track_id": "r", "role": ref_role},
        ]
    })
    s6 = _write(tmp_path / "s6.json", {"frame_index": 104, "contact_track_id": "a"})
    return s1, s5, s6


def test_full_independent_case_is_exact(tmp_path):
    s1, s5, s6 = _case_files(tmp_path)
    manifest = {
        "cases": [{
            "case_id": "x",
            "stage1": s1, "stage5": s5, "stage6": s6,
            "provenance": {"independent_gt": True, "dataset": "TEST"},
            "gt": {
                "frame_index": 104,
                "status": "VALID",
                "toucher_track_id": "a",
                "attacking_team_id": 0,
                "attack_direction_s": 1,
                "sets": {"attackers": ["a", "b"], "opponents": ["c"], "referees_excluded": ["r"]},
            },
        }]
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest_v2(str(m), str(tmp_path / "out"))
    metrics = report["summary"]["metrics_independent_gt"]
    assert metrics["full_game_state_exact_match"]["value"] == 1.0
    assert metrics["participant_set_micro_f1"]["value"] == 1.0
    assert metrics["referee_exclusion_f1"]["value"] == 1.0


def test_partial_gt_does_not_count_missing_labels_as_errors(tmp_path):
    s1, s5, s6 = _case_files(tmp_path)
    manifest = {
        "cases": [{
            "case_id": "partial",
            "stage1": s1, "stage5": s5, "stage6": s6,
            "provenance": {"independent_gt": True, "dataset": "FOOTPASS"},
            "gt": {"attacking_team_id": 0},
        }]
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest_v2(str(m), str(tmp_path / "out"))
    metrics = report["summary"]["metrics_independent_gt"]
    assert metrics["attacking_team_accuracy"]["value"] == 1.0
    assert metrics["attacking_team_accuracy"]["eligible"] == 1
    assert metrics["attack_direction_accuracy"]["value"] is None
    assert metrics["attack_direction_accuracy"]["eligible"] == 0
    assert metrics["full_game_state_exact_match"]["eligible"] == 0


def test_profile_is_incomplete_when_sample_count_is_too_small(tmp_path):
    s1, s5, s6 = _case_files(tmp_path)
    manifest = {
        "cases": [{
            "case_id": "x",
            "stage1": s1, "stage5": s5, "stage6": s6,
            "provenance": {"independent_gt": True},
            "gt": {
                "status": "VALID", "toucher_track_id": "a", "attacking_team_id": 0,
                "attack_direction_s": 1,
                "sets": {"attackers": ["a", "b"], "opponents": ["c"], "referees_excluded": ["r"]},
            },
        }]
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest_v2(str(m), str(tmp_path / "out"), profile="pilot-minimum")
    assert report["acceptance"]["status"] == "INCOMPLETE"
    assert report["acceptance"]["gates"]["toucher_track_accuracy"]["status"] == "NOT_EVALUATED"
    assert report["acceptance"]["gates"]["invariant_pass_rate"]["status"] == "PASS"


def test_referee_misclassification_is_visible_in_metrics(tmp_path):
    s1, s5, s6 = _case_files(tmp_path, ref_role="PLAYER")
    manifest = {
        "cases": [{
            "case_id": "bad-ref",
            "stage1": s1, "stage5": s5, "stage6": s6,
            "provenance": {"independent_gt": True},
            "gt": {
                "sets": {"attackers": ["a", "b"], "opponents": ["c"], "referees_excluded": ["r"]},
            },
        }]
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest_v2(str(m), str(tmp_path / "out"))
    metrics = report["summary"]["metrics_independent_gt"]
    assert metrics["referee_exclusion_exact_match"]["value"] == 0.0
    assert metrics["referee_exclusion_f1"]["value"] is None or metrics["referee_exclusion_f1"]["value"] == 0.0
    assert metrics["participant_set_micro_f1"]["value"] == 1.0
    # A referee can still be excluded from attacker/opponent sets because team_id is null;
    # the dedicated referee metric must expose the semantic failure.


def test_reference_report_and_template_have_provenance():
    refs = reference_report()
    assert len(refs["third_party_references"]) >= 5
    assert "pilot-minimum" in refs["engineering_profiles"]
    template = manifest_template()
    assert template["schema_version"] == "stage7-eval-manifest-2.0"
    assert template["cases"][0]["provenance"]["independent_gt"] is True
