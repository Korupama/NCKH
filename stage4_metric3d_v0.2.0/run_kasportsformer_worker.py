from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage4_metric3d.initializers.kasportsformer_worker import (
    KASportsFormerRunConfig,
    run_kasportsformer_worker,
)


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official KASportsFormer from Stage-3 COCO17 tracks"
    )
    parser.add_argument("--stage3-state", required=True)
    parser.add_argument("--kasportsformer-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-raw", required=True)
    parser.add_argument("--output-cache", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--window-radius", type=int, default=13)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()

    result = run_kasportsformer_worker(
        KASportsFormerRunConfig(
            stage3_state=_resolved(args.stage3_state),
            repository_root=_resolved(args.kasportsformer_root),
            model_config=_resolved(args.model_config),
            checkpoint=_resolved(args.checkpoint),
            output_raw=_resolved(args.output_raw),
            output_cache=_resolved(args.output_cache) if args.output_cache else None,
            device=args.device,
            batch_size=args.batch_size,
            window_radius_frames=args.window_radius,
            selected_only=args.selected_only,
        )
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
