from __future__ import annotations

import argparse
import json

from stage4_metric3d.backends.field_converter.sam3d_manifest import build_sam3d_manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare Stage-4 SAM3D worker manifest from Stage-3 tracks")
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--output", default="runs/stage4_v05/sam3d_worker_manifest.json")
    args = p.parse_args()
    result = build_sam3d_manifest(stage3_state=args.stage3_state, output_path=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
