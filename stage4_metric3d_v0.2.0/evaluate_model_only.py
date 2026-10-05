from __future__ import annotations

import argparse
import json

from stage4_metric3d.model_only_evaluation import evaluate_model_only_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Stage-4 model-only output without rerunning inference")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="Model-only .npz output")
    parser.add_argument("--report", required=True, help="Evaluation report JSON")
    args = parser.parse_args()
    report = evaluate_model_only_output(manifest_path=args.manifest, output_path=args.output, report_path=args.report)
    print(json.dumps({
        "status": "PASS_EVALUATION" if report["metric_computation_completed"] else "NOT_EVALUATED",
        "report": args.report,
        "coverage": report["coverage"],
        "accuracy_claim_allowed": report["accuracy_claim_allowed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
