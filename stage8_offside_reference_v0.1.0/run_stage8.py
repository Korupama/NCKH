from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage8_offside_reference.core import build_offside_reference
from stage8_offside_reference.visualization import render_longitudinal_qa


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 8 - Offside Reference Geometry")
    p.add_argument("--stage4", required=True, help="Stage 4 downstream handoff JSON")
    p.add_argument("--stage6", required=True, help="Stage 6 downstream handoff JSON")
    p.add_argument("--stage7", required=True, help="Stage 7 game_state_context.json")
    p.add_argument("--allow-partial-opponents", action="store_true", help="Allow missing/unusable opponents when at least 2 are usable")
    p.add_argument("--allow-ball-fallback", action="store_true", help="Fallback to second-last opponent when ball geometry is missing")
    p.add_argument("--allow-root-fallback", action="store_true", help="Fallback to root_world_m when legal landmarks are missing")
    p.add_argument("--output", required=True, help="offside_reference_state.json")
    p.add_argument("--plot", help="Optional longitudinal QA PNG")
    args = p.parse_args()

    state = build_offside_reference(
        args.stage4,
        args.stage6,
        args.stage7,
        allow_partial_opponents=args.allow_partial_opponents,
        allow_ball_fallback=args.allow_ball_fallback,
        allow_root_fallback=args.allow_root_fallback,
    )
    payload = state.to_dict()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.plot:
        render_longitudinal_qa(payload, args.plot)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if state.status in {"VALID", "DEGRADED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
