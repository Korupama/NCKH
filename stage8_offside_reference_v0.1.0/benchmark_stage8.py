from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage8_offside_reference.benchmark import run_oracle_benchmark
from stage8_offside_reference.evaluation import evaluate_manifest


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
        "schema_version": "stage8-eval-manifest-1.0",
        "protocol": {"split": "qa_only", "frozen": False, "notes": "Fill GT only from independent labels."},
        "cases": [{
            "case_id": "case001",
            "stage4": "relative/path/to/stage4_downstream_handoff.json",
            "stage6": "relative/path/to/stage6_downstream_handoff.json",
            "stage7": "relative/path/to/game_state_context.json",
            "provenance": {"independent_gt": False, "gt_source": None},
            "gt": {
                "status": "VALID",
                "second_last_track_id": "track_011",
                "reference_source": "SECOND_LAST_OPPONENT",
                "reference_x_m": None,
                "opponent_order": []
            }
        }]
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 8 benchmark CLI")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("oracle"); q.add_argument("--output-dir", required=True); q.set_defaults(func=cmd_oracle)
    q = sub.add_parser("evaluate"); q.add_argument("--manifest", required=True); q.add_argument("--output-dir", required=True); q.set_defaults(func=cmd_evaluate)
    q = sub.add_parser("manifest-template"); q.add_argument("--output", required=True); q.set_defaults(func=cmd_template)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
