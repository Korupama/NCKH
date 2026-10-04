from __future__ import annotations

"""Fail-closed Stage-4 release decision from the existing provenance report."""

import argparse
import json
from pathlib import Path
from typing import Any


def build_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    acceptance = report.get("acceptance") or {}
    tests_pass = acceptance.get("implementation_gate") == "PASS_IMPLEMENTATION"
    model_assets = acceptance.get("model_only_pretrained_inference") == "READY"
    metric_evaluated = acceptance.get("model_only_accuracy") == "EVALUATED"
    accuracy_allowed = bool(acceptance.get("accuracy_claim_allowed", False))
    if not tests_pass:
        decision = "REJECT_IMPLEMENTATION"
    elif not model_assets or not metric_evaluated or not accuracy_allowed:
        decision = "DEFER_ACCURACY_CLAIM"
    else:
        decision = "REVIEW_HELDOUT_NON_REGRESSION"
    return {
        "schema_version": "stage4-phase9-acceptance-1.0",
        "decision": decision,
        "implementation_gate": acceptance.get("implementation_gate"),
        "model_only_pretrained_inference": acceptance.get("model_only_pretrained_inference"),
        "model_only_accuracy": acceptance.get("model_only_accuracy"),
        "accuracy_claim_allowed": accuracy_allowed,
        "research_accuracy_frozen": bool(acceptance.get("research_accuracy_frozen", False)),
        "release_candidate": decision == "REVIEW_HELDOUT_NON_REGRESSION",
        "required_before_accuracy_claim": [
            "verified Stage-4 pretrained checkpoint/MHR/runtime",
            "authorized independent metric-GT benchmark",
            "sequence-disjoint development/calibration/holdout manifests",
            "paired direct-vs-refined metrics and difficult-slice non-regression",
            "calibration evidence if uncertainty is claimed",
        ],
        "limitations": [
            "Synthetic validation and self-consistency residuals are not metric accuracy.",
            "Missing Stage-1/3 integration artifacts do not block model-only development, but integration release is not claimed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a fail-closed Stage-4 Phase-9 acceptance decision")
    parser.add_argument("--report", type=Path, default=Path("docs/STAGE4_PHASE0_BASELINE.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/STAGE4_PHASE9_ACCEPTANCE.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    acceptance = build_acceptance(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(acceptance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
