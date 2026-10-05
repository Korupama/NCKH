from __future__ import annotations

import argparse
import json

from stage4_metric3d.model_only_systems import compare_model_only_systems


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare independent Stage-4 model-only systems on one benchmark cohort")
    parser.add_argument("--systems-manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = compare_model_only_systems(systems_manifest_path=args.systems_manifest, report_path=args.report)
    print(json.dumps({
        "status": "PASS_COMPARISON" if report["metric_computation_completed"] else "NOT_EVALUATED",
        "report": args.report,
        "systems": len(report["systems"]),
        "accuracy_claim_allowed": report["accuracy_claim_allowed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
