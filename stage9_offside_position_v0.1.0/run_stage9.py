from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage9_offside_position.adapters import load_json
from stage9_offside_position.core import build_offside_position_state
from stage9_offside_position.visualization import load_frame, render_overlay, save_overlay


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 9 - Offside Position Classification (best-effort demo first)")
    p.add_argument("--stage4", required=True)
    p.add_argument("--stage7", required=True)
    p.add_argument("--stage8", help="Optional Stage 8 state; best-effort mode can recompute a reference from Stage 4/6/7")
    p.add_argument("--stage6", help="Optional Stage 6 handoff used when Stage 8 reference is unavailable")
    p.add_argument("--output", required=True)
    p.add_argument("--epsilon-m", type=float, default=1e-9)
    p.add_argument("--strict", action="store_true", help="Honor upstream VALID status and missing-geometry gates")
    p.add_argument("--allow-defender-only", action="store_true", help="Classify positions against an available Stage 8 defender-only reference when ball/contact is missing")
    p.add_argument("--allow-tentative-context", action="store_true", help="Classify when the only Stage 7/8 limitation is tentative spatial contact")
    p.add_argument("--stage1")
    p.add_argument("--stage3")
    p.add_argument("--image")
    p.add_argument("--video")
    p.add_argument("--overlay")
    args = p.parse_args()

    state = build_offside_position_state(args.stage4, args.stage7, args.stage8, stage6_input=args.stage6, epsilon_m=args.epsilon_m, best_effort=not args.strict, allow_defender_only=args.allow_defender_only, allow_tentative_context=args.allow_tentative_context)
    payload = state.to_dict()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.overlay:
        stage1 = load_json(args.stage1) if args.stage1 else None
        stage3 = load_json(args.stage3) if args.stage3 else None
        stage4 = load_json(args.stage4)
        frame = load_frame(image_path=args.image, video_path=args.video, frame_index=int(payload.get("frame_index") or 0))
        overlay = render_overlay(frame, payload, stage4=stage4, stage3=stage3, stage1=stage1)
        save_overlay(args.overlay, overlay)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] in {"VALID", "DEGRADED", "DEMO_BEST_EFFORT", "DEMO_PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
