#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.dsp3_adapter import inspect_3dsp, run_3dsp_benchmark
from benchmark.coco_wholebody import export_predictions, official_xtcoco_eval
from benchmark.pose23 import evaluate_pose23


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-3 pose benchmark harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("inspect-3dsp")
    q.add_argument("--root", required=True, type=Path)

    q = sub.add_parser("run-3dsp")
    q.add_argument("--root", required=True, type=Path)
    q.add_argument("--split", default="train", choices=["train", "test"])
    q.add_argument("--rtmw-model", required=True, type=Path)
    q.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    q.add_argument("--max-samples", type=int, default=None)
    q.add_argument("--bbox-padding", type=float, default=1.25,
                   help="RTMW bbox padding multiplier for preprocessing ablations.")
    q.add_argument("--crop-scale", type=float, default=1.0,
                   help="Additional crop scale multiplier for preprocessing ablations.")
    q.add_argument("--crop-scales", default=None,
                   help="Comma-separated scales for QA-only multi-crop selection; always includes scale 1.0.")
    q.add_argument("--shot-manifest", type=Path, default=None,
                   help="JSON shot-level manifest; only its shot_ids are evaluated.")
    q.add_argument("--output-dir", required=True, type=Path)

    q = sub.add_parser("run-coco-wholebody")
    q.add_argument("--images-root", required=True, type=Path)
    q.add_argument("--annotations", required=True, type=Path)
    q.add_argument("--rtmw-model", required=True, type=Path)
    q.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    q.add_argument("--max-persons", type=int, default=None)
    q.add_argument("--output-dir", required=True, type=Path)
    q.add_argument("--skip-xtcoco-eval", action="store_true")

    q = sub.add_parser("eval-pose23")
    q.add_argument("--ground-truth", required=True, type=Path)
    q.add_argument("--predictions", required=True, type=Path)
    q.add_argument("--output", required=True, type=Path)

    args = p.parse_args()
    if args.cmd == "inspect-3dsp":
        print(json.dumps(inspect_3dsp(args.root), indent=2, ensure_ascii=False))
        return
    if args.cmd == "run-3dsp":
        report = run_3dsp_benchmark(
            args.root,
            args.rtmw_model,
            split=args.split,
            device=args.device,
            max_samples=args.max_samples,
            bbox_padding=args.bbox_padding,
            crop_scale=None if args.crop_scales is not None else args.crop_scale,
            crop_scales=(
                [float(value.strip()) for value in args.crop_scales.split(",") if value.strip()]
                if args.crop_scales is not None else None
            ),
            shot_manifest=args.shot_manifest,
        )
        out = args.output_dir / "3dsp_benchmark_summary.json"
        _write(out, report)
        print(json.dumps({
            "status":"COMPLETE",
            "output":str(out),
            "metrics":report["metrics"],
            "metrics_by_crop_scale":report["metrics_by_crop_scale"],
            "selected_crop_scale_counts":report["selected_crop_scale_counts"],
        }, indent=2, ensure_ascii=False))
        return
    if args.cmd == "run-coco-wholebody":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        preds = export_predictions(args.images_root, args.annotations, args.rtmw_model, device=args.device, max_persons=args.max_persons)
        pred_path = args.output_dir / "coco_wholebody_predictions.json"
        _write(pred_path, preds)
        eval_report = {"status":"SKIPPED_BY_USER"} if args.skip_xtcoco_eval else official_xtcoco_eval(args.annotations, pred_path)
        report = {
            "schema_version":"stage3-coco-wholebody-benchmark-1.0",
            "persons":len(preds), "predictions":str(pred_path), "official_xtcoco":eval_report,
            "note":"Top-down diagnostic uses GT person boxes; it isolates pose estimation from human detection."
        }
        summary_path = args.output_dir / "coco_wholebody_benchmark_summary.json"
        _write(summary_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.cmd == "eval-pose23":
        report = evaluate_pose23(args.ground_truth, args.predictions)
        _write(args.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
