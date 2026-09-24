from __future__ import annotations

"""Canonical Stage-6 benchmark entry point for the reduced offside pipeline.

Scientific protocol:
  - SoccerNet-v3D fixed test split: ball detection + single-frame Ball-X.
  - ISSIA-3D cameras 3-6: temporal/hybrid development only.
  - ISSIA-3D cameras 1-2: held-out temporal/cross-domain test.
  - Frozen contact manifest: contact-track / region / optional Ball-X at t0.

Project replay frames without independent GT remain QA/regression samples only.
"""

import argparse
import json
from pathlib import Path

from ball_localization.evaluation.benchmark_2d import run_2d_benchmark
from ball_localization.evaluation.benchmark_3d_oracle import run_3d_oracle_benchmark
from ball_localization.evaluation.benchmark_3d_e2e import run_3d_e2e_benchmark
from ball_localization.evaluation.benchmark_3d_decomposition_v12 import run_3d_center_diameter_decomposition
from ball_localization.evaluation.benchmark_ground_plane_v13 import run_ground_plane_diagnostic
from ball_localization.evaluation.contact_benchmark_v1 import (
    benchmark_contact_manifest,
    write_manifest_template,
)
from ball_localization.evaluation.contact_production_v14 import benchmark_contact_production_manifest
from ball_localization.evaluation.contact_preflight_v142 import preflight_contact_manifest
from ball_localization.evaluation.footpass_manifest_v14 import (
    build_footpass_contact_manifest,
    write_footpass_bridge_template,
)
from ball_localization.evaluation.tbd_status_v14 import build_tbd_status_report
from ball_localization.evaluation.issia3d import (
    benchmark_issia3d_temporal,
    calibrate_issia3d_hybrid,
)
from ball_localization.evaluation.protocol_v1 import (
    write_primary_protocol_report,
    write_geometry_diagnostic_report,
)
from ball_localization.version import runtime_provenance


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_primary(args) -> int:
    out = Path(args.output_dir).expanduser().resolve()
    detector_dir = out / "01_detector_2d"
    oracle_dir = out / "02_oracle_geometry"
    e2e_dir = out / "03_e2e_top1"
    report_dir = out / "04_protocol_report"
    smoke = args.max_images is not None or args.max_rows is not None

    detector = run_2d_benchmark(
        csv_path=args.csv,
        image_root=args.image_root,
        weights=args.weights,
        split=args.split,
        output_dir=detector_dir,
        conf_floor=args.conf_floor,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        imgsz=args.imgsz,
        device=args.device,
        nms_iou=args.nms_iou,
        max_images=args.max_images,
        progress_every=args.progress_every,
        failure_images=args.failure_images,
        gt_box_mode="optimized",
    )
    oracle = run_3d_oracle_benchmark(
        csv_path=args.csv,
        split=args.split,
        output_dir=oracle_dir,
        diameter_source="optimized",
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        progress_every=max(args.progress_every, 1),
        failure_images=0,
    )
    e2e = run_3d_e2e_benchmark(
        csv_path=args.csv,
        benchmark_2d_dir=detector_dir,
        split=args.split,
        output_dir=e2e_dir,
        selection="top1",
        localization_mode="size-prior",
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        progress_every=max(args.progress_every, 1),
        failure_images=args.failure_images,
        allow_partial_2d=smoke,
        gt_box_mode="optimized",
    )
    protocol = write_primary_protocol_report(
        report_dir,
        detector_report=detector,
        oracle_report=oracle,
        e2e_report=e2e,
        split=args.split,
        smoke=smoke,
    )
    _print({
        "status": protocol["status"],
        "scientific_claim_ready": protocol["scientific_claim_ready"],
        "output_dir": str(out),
        "primary_report": str(report_dir / "stage6_primary_benchmark.json"),
        "metrics": {
            "detection": protocol["stage6a_ball_detection"],
            "ball_x_oracle": protocol["stage6b_oracle_geometry"],
            "ball_x_e2e": protocol["stage6b_e2e_top1"],
        },
    })
    return 0


def cmd_geometry_diagnostics(args) -> int:
    out = Path(args.output_dir).expanduser().resolve()
    decomp_dir = out / "01_center_diameter_decomposition"
    best_iou_dir = out / "02_e2e_best_iou"
    report_dir = out / "03_report"
    smoke = args.max_rows is not None

    decomp = run_3d_center_diameter_decomposition(
        csv_path=args.csv,
        benchmark_2d_dir=args.benchmark_2d_dir,
        split=args.split,
        output_dir=decomp_dir,
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        allow_partial_2d=args.allow_partial_2d,
        gt_box_mode="optimized",
        progress_every=args.progress_every,
    )
    best_iou = run_3d_e2e_benchmark(
        csv_path=args.csv,
        benchmark_2d_dir=args.benchmark_2d_dir,
        split=args.split,
        output_dir=best_iou_dir,
        selection="best-iou",
        localization_mode="size-prior",
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        max_rows=args.max_rows,
        progress_every=max(args.progress_every, 1),
        failure_images=args.failure_images,
        allow_partial_2d=args.allow_partial_2d,
        gt_box_mode="optimized",
    )
    primary_report = None
    if args.primary_report:
        primary_report = json.loads(Path(args.primary_report).expanduser().read_text(encoding="utf-8"))
    report = write_geometry_diagnostic_report(
        report_dir,
        decomposition_report=decomp,
        best_iou_report=best_iou,
        split=args.split,
        smoke=smoke,
        primary_report=primary_report,
    )
    _print({
        "status": report["status"],
        "scientific_diagnostic_ready": report["scientific_diagnostic_ready"],
        "production_metric": report["production_metric"],
        "output_dir": str(out),
        "report": str(report_dir / "stage6_geometry_diagnostics.json"),
        "center_diameter": report["center_diameter_decomposition"],
        "best_iou": report["best_iou_diagnostic"],
    })
    return 0


def cmd_ground_plane_diagnostics(args) -> int:
    out = Path(args.output_dir).expanduser().resolve()
    report = run_ground_plane_diagnostic(
        csv_path=args.csv,
        benchmark_2d_dir=args.benchmark_2d_dir,
        split=args.split,
        output_dir=out,
        ball_radius_m=args.ball_radius_m,
        pitch_margin_m=args.pitch_margin_m,
        max_height_m=args.max_height_m,
        tolerances_m=tuple(args.ground_tolerances_m),
        primary_tolerance_m=args.primary_ground_tolerance_m,
        max_rows=args.max_rows,
        allow_partial_2d=args.allow_partial_2d,
        progress_every=args.progress_every,
    )
    primary = report["primary_near_ground_proxy"]
    _print({
        "status": report["status"],
        "scientific_diagnostic_ready": report["scientific_diagnostic_ready"],
        "production_metric": report["production_metric"],
        "output_dir": str(out),
        "report": str(out / "stage6_ground_plane_diagnostic.json"),
        "primary_near_ground_proxy": primary,
    })
    return 0


def _load_hybrid_config(path: str | None):
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return (payload.get("best") or {}).get("config") or payload.get("config")


def cmd_issia_calibrate(args) -> int:
    report = calibrate_issia3d_hybrid(
        csv_path=args.csv,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        cameras=(3, 4, 5, 6),
        fps=args.fps,
        protocol="v043-compatible",
    )
    _print({
        "status": report["status"],
        "development_cameras": [3, 4, 5, 6],
        "held_out_test_cameras": [1, 2],
        "best": report["best"],
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    })
    return 0


def cmd_issia_test(args) -> int:
    report = benchmark_issia3d_temporal(
        csv_path=args.csv,
        calibration_path=args.calibration,
        output_dir=args.output_dir,
        cameras=(1, 2),
        fps=args.fps,
        ball_radius_m=args.ball_radius_m,
        protocol="v043-compatible",
        hybrid_config=_load_hybrid_config(args.hybrid_config_json),
    )
    _print({
        "status": report["status"],
        "test_cameras": [1, 2],
        "protocol": report["protocol"],
        "methods": report["methods"],
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    })
    return 0


def cmd_contact(args) -> int:
    report = benchmark_contact_manifest(
        args.manifest,
        args.output_dir,
        allow_missing_states=args.allow_missing_states,
    )
    _print({
        "status": report["status"],
        "scientific_claim_ready": report["scientific_claim_ready"],
        "metrics": report["metrics"],
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    })
    return 0


def cmd_contact_template(args) -> int:
    path = write_manifest_template(args.output)
    _print({"status": "READY", "template": str(path)})
    return 0



def cmd_contact_preflight(args) -> int:
    report = preflight_contact_manifest(args.manifest)
    _print(report)
    return 0 if report["status"] == "READY" else 2

def cmd_contact_production(args) -> int:
    report = benchmark_contact_production_manifest(
        args.manifest, args.output_dir, allow_missing_states=args.allow_missing_states
    )
    _print({
        "status": report["status"],
        "scientific_claim_ready": report["scientific_claim_ready"],
        "metric_readiness": report["metric_readiness"],
        "metrics": report["metrics"],
        "project_target_assessment": report["project_target_assessment"],
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    })
    return 0


def cmd_footpass_bridge_template(args) -> int:
    path = write_footpass_bridge_template(args.output)
    _print({"status": "READY", "template": str(path)})
    return 0


def cmd_footpass_manifest(args) -> int:
    report = build_footpass_contact_manifest(
        playbyplay_json=args.playbyplay,
        bridge_csv=args.bridge_csv,
        output_json=args.output,
        frozen=args.frozen,
        derive_unambiguous_regions=not args.no_derive_unambiguous_regions,
    )
    _print(report)
    return 0


def cmd_tbd_status(args) -> int:
    report = build_tbd_status_report(
        output_dir=args.output_dir,
        ground_report=args.ground_report,
        issia_report=args.issia_report,
        contact_report=args.contact_report,
    )
    _print(report)
    return 0


def cmd_protocol(args) -> int:
    _print({
        "schema_version": "stage6-reduced-benchmark-protocol-1.4",
        "runtime_provenance": runtime_provenance(),
        "primary": {
            "dataset": "SoccerNet-v3D",
            "split": "test",
            "tasks": ["ball detection", "single-frame Ball-X localization", "center-vs-box-size error decomposition"],
            "metrics": [
                "AP50", "mAP50_95", "Recall@IoU0.5", "CandidateRecall@5",
                "CenterErrorPx median/P90", "Ball-X MAE/median/P90/P95", "coverage",
            ],
            "forbidden": "Do not chain SNv3D CSV rows as a temporal trajectory.",
        },
        "geometry_diagnostics": {
            "dataset": "SoccerNet-v3D",
            "requires": "completed 2D prediction cache",
            "variants": [
                "GT center + GT diameter",
                "pred center + GT diameter",
                "GT center + pred diameter",
                "pred center + pred diameter",
                "GT-assisted best-IoU proposal",
            ],
            "semantics": "diagnostic error attribution; best-IoU is not a production metric",
        },
        "ground_plane_diagnostics": {
            "dataset": "SoccerNet-v3D",
            "requires": "completed 2D prediction cache",
            "primary_proxy": "|GT_Z - ball_radius| <= 0.10 m",
            "variants": [
                "ground plane + GT center",
                "ground plane + predicted center",
                "size prior + GT bbox",
                "size prior + predicted top1 bbox",
            ],
            "semantics": "geometry proxy for the v0.5.1 FOOT branch; GT Z selects the subset only; contact association is not evaluated",
        },
        "secondary_temporal": {
            "dataset": "ISSIA-3D",
            "development": "cameras 3-6",
            "held_out_test": "cameras 1-2",
        },
        "contact": {
            "dataset": "FOOTPASS validation-derived manifest or separately frozen project set",
            "metrics": [
                "contact assigned precision", "assignment coverage", "overall contact accuracy",
                "FOOT-vs-NON_FOOT accuracy/precision/recall", "optional production Ball-X MAE/P90/P95/coverage",
            ],
            "ball_x_limitation": "FOOTPASS has actor/action GT but no independent metric ball-X; full production Ball-X requires frozen external GT.",
        },
        "readiness_aggregation": "Use tbd-status to combine ground-plane, ISSIA held-out, and contact/production reports without treating missing GT as PASS.",
        "qa": "Project-selected replay frames are regression/smoke tests only.",
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 6 reduced offside benchmark protocol v1.4")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("protocol", help="Print the frozen benchmark design")
    q.set_defaults(func=cmd_protocol)

    q = sub.add_parser("primary", help="Run SoccerNet-v3D detector + oracle + end-to-end Ball-X benchmark")
    q.add_argument("--csv", required=True, help="Official SNv3D.csv")
    q.add_argument("--image-root", required=True, help="SoccerNet Frames-v3 root; ZIP-backed layout supported")
    q.add_argument("--weights", required=True, help="Recommended: official yolo-sn-ball-opt.pt")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--split", choices=["train", "test"], default="test")
    q.add_argument("--device", default="cpu")
    q.add_argument("--imgsz", type=int, default=1920)
    q.add_argument("--conf-floor", type=float, default=0.05)
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--candidate-k", type=int, default=5)
    q.add_argument("--nms-iou", type=float, default=0.50)
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--pitch-margin-m", type=float, default=6.0)
    q.add_argument("--max-height-m", type=float, default=30.0)
    q.add_argument("--progress-every", type=int, default=10)
    q.add_argument("--failure-images", type=int, default=20)
    q.add_argument("--max-images", type=int, help="QA/smoke only; makes scientific_claim_ready=false")
    q.add_argument("--max-rows", type=int, help="QA/smoke only; makes scientific_claim_ready=false")
    q.set_defaults(func=cmd_primary)

    q = sub.add_parser("geometry-diagnostics", help="Reuse the 2D cache to decompose center-vs-diameter Ball-X error and run GT-assisted best-IoU diagnostics")
    q.add_argument("--csv", required=True, help="Official SNv3D.csv")
    q.add_argument("--benchmark-2d-dir", required=True, help="Completed primary/01_detector_2d cache; detector is not rerun")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--split", choices=["train", "test"], default="test")
    q.add_argument("--primary-report", help="Optional stage6_primary_benchmark.json for exact consistency checks")
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--pitch-margin-m", type=float, default=6.0)
    q.add_argument("--max-height-m", type=float, default=30.0)
    q.add_argument("--progress-every", type=int, default=100)
    q.add_argument("--failure-images", type=int, default=20)
    q.add_argument("--max-rows", type=int, help="QA/smoke only; full test split is required for scientific diagnostics")
    q.add_argument("--allow-partial-2d", action="store_true")
    q.set_defaults(func=cmd_geometry_diagnostics)

    q = sub.add_parser("ground-plane-diagnostics", help="Evaluate the v0.5.1 FOOT Z=ball-radius geometry on SoccerNet-v3D near-ground GT-height subsets")
    q.add_argument("--csv", required=True, help="Official SNv3D.csv")
    q.add_argument("--benchmark-2d-dir", required=True, help="Completed primary/01_detector_2d cache; detector is not rerun")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--split", choices=["train", "test"], default="test")
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--pitch-margin-m", type=float, default=6.0)
    q.add_argument("--max-height-m", type=float, default=30.0)
    q.add_argument("--ground-tolerances-m", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    q.add_argument("--primary-ground-tolerance-m", type=float, default=0.10)
    q.add_argument("--progress-every", type=int, default=100)
    q.add_argument("--max-rows", type=int, help="QA/smoke only; full test split is required for scientific diagnostics")
    q.add_argument("--allow-partial-2d", action="store_true")
    q.set_defaults(func=cmd_ground_plane_diagnostics)

    q = sub.add_parser("issia-calibrate", help="Tune hybrid selector on ISSIA-3D development cameras 3-6 only")
    q.add_argument("--csv", required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--fps", type=float, default=25.0)
    q.set_defaults(func=cmd_issia_calibrate)

    q = sub.add_parser("issia-test", help="Evaluate temporal/hybrid geometry on held-out ISSIA-3D cameras 1-2")
    q.add_argument("--csv", required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--fps", type=float, default=25.0)
    q.add_argument("--ball-radius-m", type=float, default=0.11)
    q.add_argument("--hybrid-config-json", help="Config produced by issia-calibrate on cameras 3-6")
    q.set_defaults(func=cmd_issia_test)

    q = sub.add_parser("contact", help="Evaluate Stage6 contact output against a frozen canonical manifest")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--allow-missing-states", action="store_true")
    q.set_defaults(func=cmd_contact)

    q = sub.add_parser("contact-template", help="Write a canonical contact benchmark manifest template")
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_contact_template)

    q = sub.add_parser("contact-preflight", help="Validate Stage-6 state paths and GT bridges before contact-production")
    q.add_argument("--manifest", required=True)
    q.set_defaults(func=cmd_contact_preflight)

    q = sub.add_parser("contact-production", help="Evaluate contact actor/track, FOOT-vs-NON_FOOT, and optional production Ball-X from a frozen manifest")
    q.add_argument("--manifest", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--allow-missing-states", action="store_true")
    q.set_defaults(func=cmd_contact_production)

    q = sub.add_parser("footpass-bridge-template", help="Write the CSV bridge used to bind official FOOTPASS events to Stage-6 state files")
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_footpass_bridge_template)

    q = sub.add_parser("footpass-build-manifest", help="Validate a bridge CSV against official FOOTPASS play-by-play GT and build the canonical contact manifest")
    q.add_argument("--playbyplay", required=True, help="Official FOOTPASS playbyplay_val.json")
    q.add_argument("--bridge-csv", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--frozen", action="store_true", help="Mark the resulting case selection/annotations as frozen")
    q.add_argument("--no-derive-unambiguous-regions", action="store_true", help="Do not derive HEAD for Header or ARM_HAND for Throw-in")
    q.set_defaults(func=cmd_footpass_manifest)

    q = sub.add_parser("tbd-status", help="Aggregate the remaining Stage-6 TBD benchmark reports into research/offside readiness statuses")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--ground-report", help="stage6_ground_plane_diagnostic.json")
    q.add_argument("--issia-report", help="benchmark_issia3d_temporal.json from held-out cameras 1-2")
    q.add_argument("--contact-report", help="stage6_contact_production_benchmark.json")
    q.set_defaults(func=cmd_tbd_status)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
