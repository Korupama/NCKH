from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage8_offside_reference.preflight import build_preflight


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 8 integration preflight")
    p.add_argument("--stage4", required=True)
    p.add_argument("--stage6", required=True)
    p.add_argument("--stage7", required=True)
    p.add_argument("--allow-partial-opponents", action="store_true", help="Allow missing/unusable opponents when at least 2 are usable")
    p.add_argument("--allow-ball-fallback", action="store_true", help="Fallback to second-last opponent when ball geometry is missing")
    p.add_argument("--allow-root-fallback", action="store_true", help="Fallback to root_world_m when legal landmarks are missing")
    p.add_argument("--output")
    args = p.parse_args()
    report = build_preflight(
        args.stage4,
        args.stage6,
        args.stage7,
        allow_partial_opponents=args.allow_partial_opponents,
        allow_ball_fallback=args.allow_ball_fallback,
        allow_root_fallback=args.allow_root_fallback,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] in {"READY", "DEGRADED_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
