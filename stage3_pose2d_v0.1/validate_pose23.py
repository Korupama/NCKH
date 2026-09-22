from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage3_pose2d.task_pose23 import (
    Pose23ValidationError,
    load_and_validate_manifest,
    pose23_manifest_template,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Stage-3 Pose23AtT0 manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--write-template", default=None)
    args = parser.parse_args()

    if args.write_template:
        target = Path(args.write_template).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(pose23_manifest_template(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        manifest = load_and_validate_manifest(args.manifest, allow_empty=args.allow_empty)
    except (Pose23ValidationError, json.JSONDecodeError, OSError) as exc:
        print(str(exc))
        return 1

    splits = {}
    for sample in manifest.get("samples", []):
        split = str(sample["split"])
        splits[split] = splits.get(split, 0) + 1
    print(json.dumps({"status": "VALID", "samples": len(manifest["samples"]), "split_counts": splits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
