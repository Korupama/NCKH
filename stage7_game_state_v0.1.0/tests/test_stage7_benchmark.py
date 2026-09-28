from stage7_game_state.benchmark import run_oracle_benchmark


def test_oracle_benchmark_is_perfect(tmp_path):
    report = run_oracle_benchmark(str(tmp_path))
    assert report["status"] == "PASS"
    assert report["oracle_logic_accuracy"] == 1.0


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
