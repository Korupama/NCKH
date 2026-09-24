from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage7_game_state.benchmark_v2 import (
    ENGINEERING_PROFILES,
    evaluate_manifest_v2,
    manifest_template,
    run_oracle,
    write_reference_report,
)


def _dump(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_references(args):
    report = write_reference_report(args.output_dir)
    _dump(report)
    return 0


def cmd_template(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest_template(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    return 0


def cmd_evaluate(args):
    profile = None if args.profile == "none" else args.profile
    report = evaluate_manifest_v2(args.manifest, args.output_dir, profile=profile)
    _dump(report)
    acceptance = report.get("acceptance", {}).get("status")
    return 2 if acceptance == "BELOW_TARGET" else 0


def cmd_oracle(args):
    report = run_oracle(args.output_dir)
    _dump(report)
    return 0 if report.get("status") == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 evaluation/benchmark v2")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("references", help="Write third-party comparison and internal engineering profiles")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_references)

    p = sub.add_parser("manifest-template", help="Write a v2 partial-GT manifest template")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("evaluate", help="Evaluate a frozen v2 manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--profile", choices=["none", *sorted(ENGINEERING_PROFILES)], default="none")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("oracle", help="Run deterministic oracle fixtures; target is exactly 1.0")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_oracle)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
