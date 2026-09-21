from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter

from ball_localization.datasets import SoccerNetV3DCSV, SoccerNetImageResolver
from ball_localization.evaluation import (
    compose_combined_report,
    run_2d_benchmark,
    run_cached_2d_gt_comparison,
    run_3d_e2e_benchmark,
    run_3d_oracle_benchmark,
    inspect_issia3d,
    benchmark_issia3d_temporal,
    calibrate_issia3d_hybrid,
)
from ball_localization.evaluation.common import load_json
from ball_localization.version import PACKAGE_VERSION, STAGE6_VERSION, runtime_provenance


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))



def cmd_provenance(args) -> int:
    _print(runtime_provenance())
    return 0

def cmd_inspect(args) -> int:
    _print(SoccerNetV3DCSV(args.csv).summary())
    return 0


def cmd_status(args) -> int:
    p = Path(args.output_dir) / "checkpoint.json"
    _print(load_json(p) if p.is_file() else {"status": "NOT_STARTED", "output_dir": str(Path(args.output_dir).resolve())})
    return 0

def cmd_inspect_issia3d(args) -> int:
    _print(inspect_issia3d(args.csv, args.calibration))
    return 0

def cmd_benchmark_issia3d_temporal(args) -> int:
    hybrid_config = None
    if args.hybrid_config_json:
        payload = json.loads(Path(args.hybrid_config_json).read_text(encoding="utf-8"))
        hybrid_config = (payload.get("best") or {}).get("config") or payload.get("config")
    report = benchmark_issia3d_temporal(
        csv_path=args.csv, calibration_path=args.calibration, output_dir=args.output_dir,
        cameras=args.cameras, fps=args.fps, ball_radius_m=args.ball_radius_m,
        protocol=args.protocol, hybrid_config=hybrid_config,
    )
    _print({"status": report["status"], "protocol": report["protocol"], "methods": report["methods"], "hybrid_diagnostics": report["hybrid_diagnostics"], "output_dir": str(Path(args.output_dir).resolve())})
    return 0

def cmd_calibrate_issia3d_hybrid(args) -> int:
    report = calibrate_issia3d_hybrid(
        csv_path=args.csv, calibration_path=args.calibration, output_dir=args.output_dir,
        cameras=args.cameras, fps=args.fps, protocol=args.protocol,
        ground_proxy_height_grid_m=args.ground_proxy_height_grid_m,
        ballistic_consensus_grid_m=args.ballistic_consensus_grid_m,
    )
    _print({"status": report["status"], "protocol": report["protocol"], "best": report["best"], "output_dir": str(Path(args.output_dir).resolve())})
    return 0


def cmd_inspect_images(args) -> int:
    dataset = SoccerNetV3DCSV(args.csv)
    records = dataset.split(args.split)
    if args.max_images is not None:
        records = records[: int(args.max_images)]
    resolver = SoccerNetImageResolver(args.image_root)
    counts = Counter()
    missing_examples = []
    decode_failure_examples = []
    missing_breakdown = Counter()
    for record in records:
        source = resolver.resolve_source(record)
        if source is None:
            counts["missing"] += 1
            expected_archive = resolver.expected_archive_path(record)
            reason = "archive_present_member_missing" if expected_archive.is_file() else "archive_missing"
            missing_breakdown[reason] += 1
            if len(missing_examples) < 10:
                missing_examples.append({
                    "record_id": record.record_id,
                    "expected_archive": str(expected_archive),
                    "archive_exists": bool(expected_archive.is_file()),
                    "expected_member_basename": record.basename,
                    "raw_action": record.raw.get("action"),
                    "parsed_action": bool(record.action),
                    "raw_replay": record.raw.get("replay"),
                    "is_action_frame": bool(record.is_action_frame),
                    "reason": reason,
                })
            continue
        counts[source.kind] += 1
        if args.decode:
            image, _ = resolver.read(record)
            if image is None:
                counts["decode_failed"] += 1
                if len(decode_failure_examples) < 10:
                    decode_failure_examples.append(source.reference)
    missing = int(counts.get("missing", 0))
    decode_failed = int(counts.get("decode_failed", 0))
    payload = {
        "status": "READY" if missing == 0 and decode_failed == 0 else "INCOMPLETE",
        "csv": str(Path(args.csv).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()),
        "split": args.split,
        "records": len(records),
        "image_sources": dict(counts),
        "missing_examples": missing_examples,
        "missing_breakdown": dict(missing_breakdown),
        "decode_failure_examples": decode_failure_examples,
        "zip_support": True,
        "huggingface_layout": {
            "repo_id": "SoccerNet/SoccerNet_raw_HQ",
            "revision": "frames-v3",
            "path": "<league>/<season>/<match>/Frames-v3.zip",
        },
        "filename_semantics": {
            "action": "<main_action>.png",
            "replay": "<main_action>_<replay>.png",
            "numeric_action_flags_supported": True,
            "missing_replay_implies_action_frame": True,
        },
        "note": "Primary layout is the Hugging Face frames-v3 per-match Frames-v3.zip tree; loose images remain supported.",
    }
    _print(payload)
    return 0 if payload["status"] == "READY" else 2


def cmd_run_2d(args) -> int:
    report = run_2d_benchmark(
        csv_path=args.csv,
        image_root=args.image_root,
        weights=args.weights,
        split=args.split,
        output_dir=args.output_dir,
        conf_floor=args.conf_floor,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        imgsz=args.imgsz,
        device=args.device,
        nms_iou=args.nms_iou,
        max_images=args.max_images,
        progress_every=args.progress_every,
        failure_images=args.failure_images,
        gt_box_mode=args.gt_box_mode,
    )
    _print({
        "status": report["status"],
        "evaluated_images": report["evaluated_images"],
        "missing_images": report["missing_images"],
        "image_sources": report.get("image_sources", {}),
        "gt_box_mode": report.get("gt_box_mode"),
        "metrics": report["metrics"],
        "output_dir": str(Path(args.output_dir).resolve()),
    })
    return 0


def cmd_reevaluate_2d(args) -> int:
    report = run_cached_2d_gt_comparison(
        csv_path=args.csv,
        benchmark_2d_dir=args.benchmark_2d_dir,
        split=args.split,
        output_dir=args.output_dir,
        candidate_ks=args.candidate_ks,
        iou_threshold=args.iou_threshold,
        max_images=args.max_images,
        failure_images=args.failure_images,
        allow_partial_2d=args.allow_partial_2d,
    )
    _print({
        "status": report["status"],
        "cached_records_found": report["cached_records_found"],
        "metrics": report["metrics"],
        "delta_optimized_minus_original": report["delta_optimized_minus_original"],
        "output_dir": str(Path(args.output_dir).resolve()),
    })
    return 0


def cmd_run_3d_oracle(args) -> int:
    report = run_3d_oracle_benchmark(
        csv_path=args.csv,
        split=args.split,
        output_dir=args.output_dir,
        diameter_source=args.diameter_source,
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        failure_images=args.failure_images,
    )
    _print({
        "status": report["status"],
        "diameter_source": report["diameter_source"],
        "rows": report["rows"],
        "metrics": report["metrics"],
        "output_dir": str(Path(args.output_dir).resolve()),
    })
    return 0


def cmd_run_3d_e2e(args) -> int:
    report = run_3d_e2e_benchmark(
        csv_path=args.csv,
        benchmark_2d_dir=args.benchmark_2d_dir,
        split=args.split,
        output_dir=args.output_dir,
        selection=args.selection,
        localization_mode=args.localization_mode,
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        failure_images=args.failure_images,
        allow_partial_2d=args.allow_partial_2d,
        gt_box_mode=args.gt_box_mode,
    )
    _print({
        "status": report["status"],
        "selection": report["selection"],
        "gt_box_mode": report.get("gt_box_mode"),
        "rows": report["rows"],
        "metrics": report["metrics"],
        "output_dir": str(Path(args.output_dir).resolve()),
    })
    return 0


def cmd_report(args) -> int:
    report = compose_combined_report(
        args.output_dir,
        detector_2d_dir=args.detector_2d_dir,
        oracle_original_dir=args.oracle_original_dir,
        oracle_optimized_dir=args.oracle_optimized_dir,
        e2e_dir=args.e2e_dir,
    )
    _print({
        "schema_version": report["schema_version"],
        "error_budget": report["error_budget"],
        "output_dir": str(Path(args.output_dir).resolve()),
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Stage 6 v{PACKAGE_VERSION} SoccerNet-v3D evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("provenance", help="Show the exact Stage-6 package/version imported by this CLI")
    q.set_defaults(func=cmd_provenance)

    q = sub.add_parser("inspect", help="Inspect SoccerNet-v3D CSV split counts")
    q.add_argument("--csv", required=True)
    q.set_defaults(func=cmd_inspect)

    q = sub.add_parser("status", help="Read benchmark checkpoint without loading any model")
    q.add_argument("--output-dir", required=True)
    q.set_defaults(func=cmd_status)

    q = sub.add_parser("inspect-images", help="Check SoccerNet-v3 loose/Frames-v3.zip image availability without loading YOLO")
    q.add_argument("--csv", required=True)
    q.add_argument("--image-root", required=True)
    q.add_argument("--split", default="test", choices=["train", "test"])
    q.add_argument("--max-images", type=int)
    q.add_argument("--decode", action="store_true", help="Also decode every resolved image to catch corrupt ZIP/image entries")
    q.set_defaults(func=cmd_inspect_images)

    q = sub.add_parser("inspect-issia3d", help="Preflight public ISSIA-3D oracle geometry assets")
    q.add_argument("--csv", required=True)
    q.add_argument("--calibration", required=True)
    q.set_defaults(func=cmd_inspect_issia3d)

    q = sub.add_parser("benchmark-issia3d-temporal", help="Oracle-2D ISSIA-3D comparison: monocular, temporal, and ballistic geometry")
    q.add_argument("--csv", required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--cameras", type=int, nargs="+", default=[1, 2])
    q.add_argument("--fps", type=float, default=25.0)
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--protocol", choices=["v043-compatible"], default="v043-compatible")
    q.add_argument("--hybrid-config-json", help="Frozen hybrid_calibration.json; never use camera 1–2 to create it")
    q.set_defaults(func=cmd_benchmark_issia3d_temporal)

    q = sub.add_parser("calibrate-issia3d-hybrid", help="Tune hybrid selector only on ISSIA cameras 3–6")
    q.add_argument("--csv", required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--cameras", type=int, nargs="+", default=[3, 4, 5, 6])
    q.add_argument("--fps", type=float, default=25.0)
    q.add_argument("--protocol", choices=["v043-compatible"], default="v043-compatible")
    q.add_argument("--ground-proxy-height-grid-m", type=float, nargs="+", default=[0.35, 0.50, 0.65, 0.80])
    q.add_argument("--ballistic-consensus-grid-m", type=float, nargs="+", default=[1.5, 2.5, 3.5, 5.0])
    q.set_defaults(func=cmd_calibrate_issia3d_hybrid)

    q = sub.add_parser("run-2d", help="Checkpointed YOLO ball detector benchmark")
    q.add_argument("--csv", required=True)
    q.add_argument("--image-root", required=True)
    q.add_argument("--weights", required=True)
    q.add_argument("--split", default="test", choices=["train", "test"])
    q.add_argument("--output-dir", required=True)
    q.add_argument("--conf-floor", type=float, default=0.05)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--candidate-k", type=int, default=5)
    q.add_argument("--imgsz", type=int, default=1920)
    q.add_argument("--device", default="cpu")
    q.add_argument("--nms-iou", type=float, default=0.50)
    q.add_argument("--max-images", type=int)
    q.add_argument("--progress-every", type=int, default=10)
    q.add_argument("--failure-images", type=int, default=20)
    q.add_argument(
        "--gt-box-mode", choices=["original", "optimized"], default="optimized",
        help="2D GT convention. Use optimized with yolo-sn-ball-opt.pt.",
    )
    q.set_defaults(func=cmd_run_2d)

    q = sub.add_parser("reevaluate-2d", help="Re-score an existing detector cache against original and optimized GT without rerunning YOLO")
    q.add_argument("--csv", required=True)
    q.add_argument("--benchmark-2d-dir", required=True, help="Existing run-2d output containing predictions/*.json")
    q.add_argument("--split", default="test", choices=["train", "test"])
    q.add_argument("--output-dir", required=True)
    q.add_argument("--candidate-ks", type=int, nargs="+", default=[1, 5, 10])
    q.add_argument("--iou-threshold", type=float, default=0.50)
    q.add_argument("--max-images", type=int)
    q.add_argument("--failure-images", type=int, default=0)
    q.add_argument("--allow-partial-2d", action="store_true")
    q.set_defaults(func=cmd_reevaluate_2d)

    q = sub.add_parser("run-3d-oracle", help="Checkpointed oracle-bbox size-prior geometry benchmark")
    q.add_argument("--csv", required=True)
    q.add_argument("--split", default="test", choices=["train", "test"])
    q.add_argument("--output-dir", required=True)
    q.add_argument("--diameter-source", choices=["original", "optimized"], default="optimized")
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--pitch-margin-m", type=float, default=6.0)
    q.add_argument("--max-height-m", type=float, default=30.0)
    q.add_argument("--max-rows", type=int)
    q.add_argument("--progress-every", type=int, default=100)
    q.add_argument("--failure-images", type=int, default=0)
    q.set_defaults(func=cmd_run_3d_oracle)

    q = sub.add_parser("run-3d-e2e", help="Detector-cache -> monocular 3D benchmark")
    q.add_argument("--csv", required=True)
    q.add_argument("--benchmark-2d-dir", required=True, help="Completed run-2d output directory")
    q.add_argument("--split", default="test", choices=["train", "test"])
    q.add_argument("--output-dir", required=True)
    q.add_argument("--selection", choices=["top1", "best-iou"], default="top1")
    q.add_argument("--gt-box-mode", choices=["original", "optimized"], default="optimized", help="GT box convention used for best-iou diagnostics and reported 2D overlap")
    q.add_argument("--localization-mode", choices=["size-prior", "ground-first", "ground-only"], default="size-prior")
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--pitch-margin-m", type=float, default=6.0)
    q.add_argument("--max-height-m", type=float, default=30.0)
    q.add_argument("--max-rows", type=int)
    q.add_argument("--progress-every", type=int, default=100)
    q.add_argument("--failure-images", type=int, default=20)
    q.add_argument("--allow-partial-2d", action="store_true")
    q.set_defaults(func=cmd_run_3d_e2e)

    q = sub.add_parser("report", help="Combine detector/oracle/e2e summaries into one error-budget report")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--detector-2d-dir")
    q.add_argument("--oracle-original-dir")
    q.add_argument("--oracle-optimized-dir")
    q.add_argument("--e2e-dir")
    q.set_defaults(func=cmd_report)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Progress is checkpointed. Rerun the same command to resume.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
