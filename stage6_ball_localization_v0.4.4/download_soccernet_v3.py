from __future__ import annotations

import argparse
import json

from ball_localization.datasets.soccernet_frames_v3 import (
    HF_REPO_ID,
    HF_REVISION,
    download_frames_v3_archives,
    summarize_frames_v3_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively download the SoccerNet_raw_HQ frames-v3 per-match Frames-v3.zip archives "
            "referenced by an SNv3D.csv split. Archives stay compressed."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to SNv3D.csv")
    parser.add_argument("--output-root", required=True, help=r"Local SoccerNet root, e.g. D:\SoccerNet")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--max-games", type=int, help="Optional small probe; downloads only the first N unique matches")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--token-env",
        help="Optional environment variable containing an HF token. Omit to use the token from `hf auth login`.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the exact remote archives needed; download nothing")
    args = parser.parse_args()

    if args.dry_run:
        payload = summarize_frames_v3_plan(
            args.csv,
            args.output_root,
            split=args.split,
            max_games=args.max_games,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    try:
        payload = download_frames_v3_archives(
            args.csv,
            args.output_root,
            split=args.split,
            repo_id=args.repo_id,
            revision=args.revision,
            max_games=args.max_games,
            max_workers=args.max_workers,
            token_env=args.token_env,
        )
    except (RuntimeError, PermissionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] == "COMPLETE":
        print("Keep Frames-v3.zip compressed; Stage 6 reads the PNG members directly.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
