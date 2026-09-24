from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage7_game_state.benchmark import evaluate_manifest, run_oracle_benchmark


def cmd_oracle(args):
    report = run_oracle_benchmark(args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


def cmd_evaluate(args):
    report = evaluate_manifest(args.manifest, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_template(args):
    payload = {
        "cases": [
            {
                "case_id": "case001",
                "stage1": "relative/path/to/stage1.json",
                "stage5": "relative/path/to/stage5.json",
                "stage6": "relative/path/to/stage6.json",
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
            }
        ]
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 benchmark CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("oracle", help="Run deterministic oracle-input fixtures")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("evaluate", help="Evaluate a frozen manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("manifest-template", help="Create evaluation manifest template")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_template)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
