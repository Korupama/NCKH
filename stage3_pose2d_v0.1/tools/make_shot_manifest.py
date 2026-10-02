#!/usr/bin/env python3
"""Create a deterministic shot-level internal holdout manifest for 3DSP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def make_manifest(root: Path, split: str, seed: int, holdout_ratio: float) -> dict:
    split_root = (root / split).resolve()
    shot_ids = sorted(p.name for p in split_root.iterdir() if p.is_dir())
    ordered = sorted(shot_ids, key=lambda shot: hashlib.sha256(f"{seed}:{shot}".encode()).hexdigest())
    holdout_n = max(1, int(round(len(ordered) * holdout_ratio)))
    holdout = sorted(ordered[:holdout_n])
    development = sorted(ordered[holdout_n:])
    return {
        "schema_version": "stage3-3dsp-shot-manifest-1.0",
        "dataset": "3D Shot Posture Dataset (3DSP)",
        "split": split,
        "unit": "shot_id",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "purpose": "internal_shot_holdout; exploratory because train aggregate was previously observed",
        "development_shots": development,
        "holdout_shots": holdout,
        "shot_ids": holdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--holdout-ratio", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_manifest(args.root, args.split, args.seed, args.holdout_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "development": len(result["development_shots"]), "holdout": len(result["holdout_shots"])}, indent=2))


if __name__ == "__main__":
    main()
