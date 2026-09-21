from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage4_metric3d.stage3_adapter import load_stage3_state
from stage4_metric3d.stage3_bridge_qa import audit_stage3_bridge


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Stage 3 -> Stage 4 selected-frame QA")
    parser.add_argument("--stage3-state", required=True, help="tracked_pose_2d_state.json")
    parser.add_argument("--output", required=True, help="stage3_bridge_qa.json")
    parser.add_argument("--edge-margin-px", type=float, default=16.0)
    parser.add_argument("--crowded-iou-threshold", type=float, default=0.15)
    args = parser.parse_args()

    report = audit_stage3_bridge(
        load_stage3_state(args.stage3_state),
        edge_margin_px=args.edge_margin_px,
        crowded_iou_threshold=args.crowded_iou_threshold,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
