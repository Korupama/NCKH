from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage7_game_state.core import build_game_state_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 7 - Game-State Context")
    parser.add_argument("--stage1", required=True, help="Stage 1 JSON")
    parser.add_argument("--stage5", required=True, help="Stage 5 JSON")
    parser.add_argument("--stage6", required=True, help="Stage 6 JSON")
    parser.add_argument("--output", required=True, help="game_state_context.json")
    parser.add_argument("--allow-goalkeeper-fallback", action="store_true", help="Infer defending team from a single affiliated visible goalkeeper when contact is missing")
    args = parser.parse_args()

    ctx = build_game_state_context(args.stage1, args.stage5, args.stage6, allow_goalkeeper_fallback=args.allow_goalkeeper_fallback)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False))
    return 0 if ctx.status == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
