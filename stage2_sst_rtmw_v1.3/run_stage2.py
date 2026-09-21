#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2_entities import ReplayContext, replay_context_from_stage1_workspace, extract_analysis_window
from stage2_entities.perception import SSTRTMWStage2Backend
from stage2_entities.legacy_adapter import build_manifest_from_legacy_jsons
from stage2_entities.pipeline import finalize_stage2


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2: SST human detection/roles + pre-pose consolidation + RTMW cue cache + target-anchored tracking"
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage1-root", help="Extracted Stage-1 v12 workspace")
    source.add_argument("--replay-context", help="Canonical ReplayContext JSON")
    ap.add_argument("--video", help="Original replay override for --stage1-root")
    ap.add_argument("--output-dir", default="runs/stage2_sst_rtmw")

    perception = ap.add_mutually_exclusive_group(required=False)
    perception.add_argument("--perception-manifest", help="Reuse an existing Stage-2 perception manifest")
    perception.add_argument("--legacy-json-dir", help="Migration/testing: directory of old *_sst_pose.json files")

    ap.add_argument("--sst-checkpoint")
    ap.add_argument("--rtmw-model")
    ap.add_argument("--sst-device", choices=("auto","cpu","cuda"), default="auto")
    ap.add_argument("--rtmw-device", choices=("cpu","cuda"), default="cpu")
    ap.add_argument("--rtmw-input-width", type=int, default=288)
    ap.add_argument("--rtmw-input-height", type=int, default=384)
    ap.add_argument("--rtmw-bbox-padding", type=float, default=1.25)
    ap.add_argument("--player-threshold", type=float, default=0.50)
    ap.add_argument("--goalkeeper-threshold", type=float, default=0.50)
    ap.add_argument("--ball-threshold", type=float, default=0.35)
    ap.add_argument("--referee-threshold", type=float, default=0.55)
    ap.add_argument("--staff-threshold", type=float, default=0.60)
    ap.add_argument("--person-nms-iou", type=float, default=0.65)
    ap.add_argument("--ball-nms-iou", type=float, default=0.30)
    ap.add_argument("--cross-class-iou", type=float, default=None,
                    help="Legacy direct-IoU override for cross-class consolidation high threshold")
    ap.add_argument("--consolidation-high-iou", type=float, default=0.82)
    ap.add_argument("--consolidation-low-iou", type=float, default=0.60)
    ap.add_argument("--consolidation-max-center-distance", type=float, default=0.18)
    ap.add_argument("--consolidation-min-area-similarity", type=float, default=0.65)
    ap.add_argument("--consolidation-min-intersection-over-min", type=float, default=0.75)
    ap.add_argument("--raw-human-score-floor", type=float, default=0.20,
                    help="Persist SST human detections down to this score for threshold sweeps/rescue")
    ap.add_argument("--rescue-overlap-with-active-iou", type=float, default=0.65)
    ap.add_argument("--role-ambiguous-margin", type=float, default=0.05)
    ap.add_argument("--keypoint-threshold", type=float, default=1.0,
                    help="Raw RTMW SimCC score threshold; NOT a calibrated probability")
    ap.add_argument("--overwrite-perception", action="store_true")
    ap.add_argument("--max-assignment-cost", type=float, default=0.92)
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--rescue-max-assignment-cost", type=float, default=0.78)
    ap.add_argument("--no-temporal-rescue", action="store_true",
                    help="Disable rescue-only low-score SST observations")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.stage1_root:
        context = replay_context_from_stage1_workspace(args.stage1_root, video_override=args.video)
        context.save_json(out / "replay_context.from_stage1.json")
    else:
        context = ReplayContext.from_json(args.replay_context)

    if args.perception_manifest:
        manifest: str | Path = Path(args.perception_manifest).expanduser().resolve()
    elif args.legacy_json_dir:
        files = sorted(Path(args.legacy_json_dir).expanduser().resolve().glob("*_sst_pose.json"))
        if not files:
            raise FileNotFoundError(f"No *_sst_pose.json in {args.legacy_json_dir}")
        manifest = build_manifest_from_legacy_jsons(
            files, out / "legacy_migration",
            cross_class_iou_threshold=(args.consolidation_high_iou if args.cross_class_iou is None else args.cross_class_iou),
            ambiguous_margin=args.role_ambiguous_margin,
        )
    else:
        if not args.sst_checkpoint or not args.rtmw_model:
            ap.error("Production inference requires --sst-checkpoint and --rtmw-model unless a perception manifest is reused")
        frame_map = extract_analysis_window(context, out / "input_window_frames")
        backend = SSTRTMWStage2Backend(
            sst_checkpoint=args.sst_checkpoint,
            rtmw_model=args.rtmw_model,
            sst_device=args.sst_device,
            rtmw_device=args.rtmw_device,
            rtmw_input_width=args.rtmw_input_width,
            rtmw_input_height=args.rtmw_input_height,
            rtmw_bbox_padding=args.rtmw_bbox_padding,
            player_threshold=args.player_threshold,
            goalkeeper_threshold=args.goalkeeper_threshold,
            ball_threshold=args.ball_threshold,
            referee_threshold=args.referee_threshold,
            staff_threshold=args.staff_threshold,
            person_nms_iou=args.person_nms_iou,
            ball_nms_iou=args.ball_nms_iou,
            cross_class_iou_threshold=args.cross_class_iou,
            consolidation_high_iou=args.consolidation_high_iou,
            consolidation_low_iou=args.consolidation_low_iou,
            consolidation_max_center_distance=args.consolidation_max_center_distance,
            consolidation_min_area_similarity=args.consolidation_min_area_similarity,
            consolidation_min_intersection_over_min=args.consolidation_min_intersection_over_min,
            raw_human_score_floor=args.raw_human_score_floor,
            rescue_overlap_with_active_iou=args.rescue_overlap_with_active_iou,
            role_ambiguous_margin=args.role_ambiguous_margin,
            keypoint_threshold=args.keypoint_threshold,
        )
        manifest = backend.process_window(frame_map, out / "perception", overwrite=args.overwrite_perception)

    state = finalize_stage2(
        context=context,
        perception_manifest=manifest,
        output_dir=out,
        render_video=not args.no_video,
        max_assignment_cost=args.max_assignment_cost,
        max_gap=args.max_gap,
        use_temporal_rescue=not args.no_temporal_rescue,
        rescue_max_assignment_cost=args.rescue_max_assignment_cost,
    )
    print(json.dumps({
        "status": state.status,
        "tracks": len(state.tracks),
        "candidate_tracks": len(state.stage3_handoff.get("candidate_track_ids", [])),
        "entity_track_state": state.artifacts.get("entity_track_state"),
        "stage3_handoff": state.artifacts.get("stage3_handoff"),
        "rtmw_track_cache": state.artifacts.get("rtmw_track_cache"),
    }, indent=2))
    return 0 if state.status != "INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
