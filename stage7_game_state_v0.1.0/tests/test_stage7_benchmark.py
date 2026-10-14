from stage7_game_state.benchmark import run_oracle_benchmark


def test_fixed_policy_reports_conflicts_with_legacy_oracle(tmp_path):
    report = run_oracle_benchmark(str(tmp_path))
    # Preserve the old oracle labels: they place an attacking team's GK in ATT.
    # The new fixed policy must expose that incompatibility, not rewrite GT.
    assert report["status"] == "FAIL"
    assert {r['fixture'] for r in report['rows'] if not r['pass']} == {'right_half_team0', 'left_half_team1'}
    assert report["oracle_logic_accuracy"] == 0.6


def test_manifest_evaluator_on_example(tmp_path):
    import json
    from pathlib import Path
    from stage7_game_state.benchmark import evaluate_manifest

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "cases": [{
            "case_id": "example",
            "stage1": str(root / "examples" / "stage1.json"),
            "stage5": str(root / "examples" / "stage5.json"),
            "stage6": str(root / "examples" / "stage6.json"),
            "gt": {
                "status": "VALID",
                "attacking_team_id": 0,
                "attack_direction_s": 1,
                "sets": {
                    "attackers": ["003", "005"],
                    "opponents": ["007", "011"],
                    "referees_excluded": ["018"]
                }
            }
        }]
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest), encoding="utf-8")
    report = evaluate_manifest(str(m), str(tmp_path / "out"))
    assert report["metrics"]["full_game_state_exact_match"] == 1.0
    assert report["metrics"]["invariant_pass_rate"] == 1.0
