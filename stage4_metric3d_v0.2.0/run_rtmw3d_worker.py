from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage4_metric3d.initializers.rtmw3d import generate_rtmw3d_cache


def main() -> int:
    p = argparse.ArgumentParser(description="Generate Stage-4 RTMW3D relative-pose initializer cache")
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--model-config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--window-radius", type=int, default=13)
    p.add_argument(
        "--mmpose-root",
        default=None,
        help="Path to official MMPose v1.3.2 source tree; automatically exposes projects/rtmpose3d",
    )
    p.add_argument("--selected-only", action="store_true")
    args = p.parse_args()

    out = generate_rtmw3d_cache(
        stage3_state=args.stage3_state,
        video_path=args.video,
        output_jsonl=args.output,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        device=args.device,
        window_radius_frames=args.window_radius,
        selected_only=args.selected_only,
        mmpose_root=args.mmpose_root,
    )
    print(json.dumps({"initializer_cache": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
