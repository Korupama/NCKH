#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_entities.benchmark.soccernet_gsr import SoccerNetGSRDataset
from stage2_entities.benchmark.protocols import BenchmarkProtocol, build_protocol_windows
from stage2_entities.benchmark.runner import BenchmarkConfig, run_benchmark
from stage2_entities.benchmark.checkpointing import read_checkpoint_status
from stage2_entities.benchmark.threshold_sweep import parse_float_grid, run_threshold_sweep


def parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description="Stage-2 t0-centric SoccerNet-GSR benchmark")
    sub=ap.add_subparsers(dest="command",required=True)
    ins=sub.add_parser("inspect",help="Validate SoccerNet-GSR root/schema without running models")
    ins.add_argument("--soccernet-root",required=True)
    ins.add_argument("--split",default="valid")
    ins.add_argument("--max-sequences",type=int,default=5)
    ins.add_argument("--allow-old-version",action="store_true")

    plan=sub.add_parser("plan",help="Print deterministic t0 windows/frame budget without model inference")
    plan.add_argument("--soccernet-root",required=True)
    plan.add_argument("--split",default="valid")
    plan.add_argument("--protocol",choices=("quick","full"),default="quick")
    plan.add_argument("--window-seconds",type=float,default=1.0)
    plan.add_argument("--max-sequences",type=int)
    plan.add_argument("--sequence-id",action="append",dest="sequence_ids")
    plan.add_argument("--allow-old-version",action="store_true")

    status=sub.add_parser("status",help="Show saved benchmark checkpoint/progress without running models")
    status.add_argument("--output-dir",default="benchmark_results/stage2_quick")

    sweep=sub.add_parser("sweep-thresholds",help="Sweep Player/GK thresholds from cached v1.3 low-score SST outputs")
    sweep.add_argument("--soccernet-root",required=True)
    sweep.add_argument("--split",default="valid")
    sweep.add_argument("--benchmark-dir",required=True,help="Existing v1.3 benchmark output containing perception caches/protocol_windows.json")
    sweep.add_argument("--player-thresholds",default="0.30,0.35,0.40,0.45,0.50")
    sweep.add_argument("--goalkeeper-thresholds",default="0.30,0.35,0.40,0.45,0.50")
    sweep.add_argument("--referee-threshold",type=float,default=0.55)
    sweep.add_argument("--staff-threshold",type=float,default=0.60)
    sweep.add_argument("--output-dir")
    sweep.add_argument("--quiet",action="store_true")

    run=sub.add_parser("run",help="Run/reuse perception cache and evaluate Stage 2")
    run.add_argument("--soccernet-root",required=True)
    run.add_argument("--split",default="valid")
    run.add_argument("--protocol",choices=("quick","full"),default="quick")
    run.add_argument("--window-seconds",type=float,default=1.0,help="Half-window around t0; default ±1 s")
    run.add_argument("--max-sequences",type=int)
    run.add_argument("--sequence-id",action="append",dest="sequence_ids",help="Repeat to select explicit sequences")
    run.add_argument("--sst-checkpoint")
    run.add_argument("--rtmw-model")
    run.add_argument("--sst-device",choices=("auto","cpu","cuda"),default="auto")
    run.add_argument("--rtmw-device",choices=("cpu","cuda"),default="cpu")
    run.add_argument("--output-dir",default="benchmark_results/stage2_quick")
    run.add_argument("--overwrite-perception",action="store_true")
    run.add_argument("--allow-old-version",action="store_true")
    run.add_argument("--experiments",default="primary,no_pose,oracle_boxes_geom",
                     help="Comma separated: primary,no_pose,oracle_boxes_geom,oracle_boxes_pose")
    run.add_argument("--player-threshold",type=float,default=0.50)
    run.add_argument("--goalkeeper-threshold",type=float,default=0.50)
    run.add_argument("--referee-threshold",type=float,default=0.55)
    run.add_argument("--staff-threshold",type=float,default=0.60)
    run.add_argument("--ball-threshold",type=float,default=0.35)
    run.add_argument("--person-nms-iou",type=float,default=0.65)
    run.add_argument("--ball-nms-iou",type=float,default=0.30)
    run.add_argument("--max-assignment-cost",type=float,default=0.92)
    run.add_argument("--max-gap",type=int,default=6)
    run.add_argument("--rescue-max-assignment-cost",type=float,default=0.78)
    run.add_argument("--no-temporal-rescue",action="store_true")
    run.add_argument("--raw-human-score-floor",type=float,default=0.20)
    run.add_argument("--consolidation-high-iou",type=float,default=0.82)
    run.add_argument("--consolidation-low-iou",type=float,default=0.60)
    run.add_argument("--consolidation-max-center-distance",type=float,default=0.18)
    run.add_argument("--consolidation-min-area-similarity",type=float,default=0.65)
    run.add_argument("--consolidation-min-intersection-over-min",type=float,default=0.75)
    run.add_argument("--rescue-overlap-with-active-iou",type=float,default=0.65)
    run.add_argument("--failure-images",type=int,default=20,help="Save up to N GT/pred overlays for failure cases")
    run.add_argument("--restart-evaluation",action="store_true",
                     help="Discard evaluation checkpoints but KEEP expensive perception caches")
    run.add_argument("--progress-every",type=int,default=10,
                     help="Print perception progress every N frames (evaluation progress prints every task)")
    run.add_argument("--quiet",action="store_true",help="Suppress live progress lines; final JSON is still printed")
    return ap


def main() -> int:
    args=parser().parse_args()
    if args.command=="inspect":
        ds=SoccerNetGSRDataset(args.soccernet_root,args.split,require_version_13=not args.allow_old_version)
        print(json.dumps(ds.inspect(args.max_sequences),indent=2,ensure_ascii=False)); return 0
    if args.command=="status":
        print(json.dumps(read_checkpoint_status(args.output_dir),indent=2,ensure_ascii=False)); return 0
    if args.command=="plan":
        ds=SoccerNetGSRDataset(args.soccernet_root,args.split,require_version_13=not args.allow_old_version)
        protocol=BenchmarkProtocol.named(args.protocol,half_window_seconds=args.window_seconds)
        ids=args.sequence_ids or ds.discover()
        limit=args.max_sequences if args.max_sequences is not None else protocol.max_sequences
        if limit is not None: ids=ids[:int(limit)]
        sequences=[]; total_windows=0; total_required=0
        for sid in ids:
            seq=ds.load(sid); windows=build_protocol_windows(seq,protocol)
            required=sorted({fi for w in windows for fi in w.frame_indices})
            total_windows += len(windows); total_required += len(required)
            sequences.append({
                "sequence_id":sid,"fps":seq.fps,"frames":seq.num_frames,"version":seq.version,
                "target_frames":[w.target_frame for w in windows],
                "windows":[w.to_dict() for w in windows],"unique_inference_frames":len(required),
            })
        print(json.dumps({
            "protocol":protocol.name,"split":args.split,"sequences":len(ids),
            "windows":total_windows,"unique_inference_frames_sum":total_required,
            "items":sequences,
        },indent=2,ensure_ascii=False)); return 0
    if args.command=="sweep-thresholds":
        report=run_threshold_sweep(
            dataset_root=args.soccernet_root, split=args.split, benchmark_dir=args.benchmark_dir,
            player_thresholds=parse_float_grid(args.player_thresholds),
            goalkeeper_thresholds=parse_float_grid(args.goalkeeper_thresholds),
            referee_threshold=args.referee_threshold, staff_threshold=args.staff_threshold,
            output_dir=args.output_dir, quiet=args.quiet,
        )
        print(json.dumps({
            "recommended":report.get("recommended"),
            "output_dir":str(Path(args.output_dir).resolve()) if args.output_dir else str((Path(args.benchmark_dir).resolve()/"threshold_sweep")),
        },indent=2,ensure_ascii=False))
        return 0
    protocol=BenchmarkProtocol.named(args.protocol,half_window_seconds=args.window_seconds)
    exps=tuple(x.strip() for x in args.experiments.split(",") if x.strip())
    allowed={"primary","no_pose","oracle_boxes_geom","oracle_boxes_pose"}
    unknown=set(exps)-allowed
    if unknown: raise SystemExit(f"Unknown experiments: {sorted(unknown)}; allowed={sorted(allowed)}")
    if "primary" not in exps: raise SystemExit("primary experiment is required")
    cfg=BenchmarkConfig(
        dataset_root=Path(args.soccernet_root),split=args.split,output_dir=Path(args.output_dir),protocol=protocol,
        sequence_ids=args.sequence_ids,max_sequences=args.max_sequences,
        sst_checkpoint=Path(args.sst_checkpoint).resolve() if args.sst_checkpoint else None,
        rtmw_model=Path(args.rtmw_model).resolve() if args.rtmw_model else None,
        sst_device=args.sst_device,rtmw_device=args.rtmw_device,overwrite_perception=args.overwrite_perception,
        require_version_13=not args.allow_old_version,experiments=exps,
        player_threshold=args.player_threshold, goalkeeper_threshold=args.goalkeeper_threshold,
        referee_threshold=args.referee_threshold, staff_threshold=args.staff_threshold,
        ball_threshold=args.ball_threshold, person_nms_iou=args.person_nms_iou, ball_nms_iou=args.ball_nms_iou,
        max_assignment_cost=args.max_assignment_cost,max_gap=args.max_gap,
        use_temporal_rescue=not args.no_temporal_rescue,
        rescue_max_assignment_cost=args.rescue_max_assignment_cost,
        raw_human_score_floor=args.raw_human_score_floor,
        consolidation_high_iou=args.consolidation_high_iou,
        consolidation_low_iou=args.consolidation_low_iou,
        consolidation_max_center_distance=args.consolidation_max_center_distance,
        consolidation_min_area_similarity=args.consolidation_min_area_similarity,
        consolidation_min_intersection_over_min=args.consolidation_min_intersection_over_min,
        rescue_overlap_with_active_iou=args.rescue_overlap_with_active_iou,
        failure_images=args.failure_images,
        restart_evaluation=args.restart_evaluation,progress_every=args.progress_every,quiet=args.quiet,
    )
    try:
        summary=run_benchmark(cfg)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Progress was checkpointed. Re-run the same command to resume.")
        return 130
    print(json.dumps({
        "protocol":summary["protocol"],
        "CandidatePrecision":summary["detection"].get("CandidatePrecision"),
        "CandidateRecall":summary["detection"]["CandidateRecall"],
        "RefereeLeakageRate":summary["detection"]["RefereeLeakageRate"],"RoleMacroF1":summary["detection"]["role_macro_f1_supported"],
        "HOTA_Candidate":summary["tracking_candidate"]["HOTA"],"AssA_Candidate":summary["tracking_candidate"]["AssA"],
        "IDF1_Candidate":summary["tracking_candidate"]["IDF1"],"TCR":summary["tracking_candidate"]["TCR"],
        "AnchorCoverage":summary["tracking_candidate"].get("AnchorCoverage"),
        "ConditionalTCR":summary["tracking_candidate"].get("ConditionalTCR"),
        "acceptance_gate_passed":summary["acceptance_gate"]["passed"],"output_dir":str(cfg.output_dir.resolve()),
    },indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
