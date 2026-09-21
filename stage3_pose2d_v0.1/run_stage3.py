#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage3_pose2d import Stage3Config, run_stage3
from stage3_pose2d.stage2_adapter import load_stage2_bundle


def _build_config(args: argparse.Namespace) -> Stage3Config:
    return Stage3Config(
        emit_temporal_estimates=bool(args.emit_temporal_estimates),
        enable_fallback_reinference=bool(args.fallback_reinfer),
        fallback_crop_scales=[float(x) for x in args.fallback_crop_scales.split(",") if x.strip()],
        rtmw_model=args.rtmw_model,
        rtmw_device=args.rtmw_device,
        rtmw_input_width=args.rtmw_input_width,
        rtmw_input_height=args.rtmw_input_height,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 - tracked 2D COCO-WholeBody133 normalization and quality assurance."
    )
    parser.add_argument("--stage2-dir", type=Path, help="Directory containing the three Stage-2 handoff JSON files.")
    parser.add_argument("--entity-state", type=Path, help="Explicit stage2_entity_tracks.json path.")
    parser.add_argument("--rtmw-cache", type=Path, help="Explicit stage2_rtmw_track_cache.json path.")
    parser.add_argument("--stage3-handoff", type=Path, help="Explicit stage3_handoff.json path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", help="Validate the Stage2->Stage3 contract and exit.")
    parser.add_argument("--emit-temporal-estimates", action="store_true",
                        help="Store interpolation estimates for missing keypoints in a separate field; raw x/y are never changed.")
    parser.add_argument("--fallback-reinfer", action="store_true",
                        help="Controlled RTMW-L re-inference for missing/degraded/rejected observations.")
    parser.add_argument("--fallback-crop-scales", default="1.0,1.1,1.2")
    parser.add_argument("--rtmw-model", type=str, default=None, help="RTMW WholeBody ONNX; required only with --fallback-reinfer.")
    parser.add_argument("--rtmw-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--rtmw-input-width", type=int, default=288)
    parser.add_argument("--rtmw-input-height", type=int, default=384)
    args = parser.parse_args()

    bundle = load_stage2_bundle(
        stage2_dir=args.stage2_dir,
        entity_state=args.entity_state,
        rtmw_cache=args.rtmw_cache,
        handoff=args.stage3_handoff,
        strict=True,
    )
    if args.preflight_only:
        print(json.dumps(bundle.validation, indent=2, ensure_ascii=False))
        return

    state = run_stage3(
        stage2_dir=args.stage2_dir,
        entity_state=args.entity_state,
        rtmw_cache=args.rtmw_cache,
        handoff=args.stage3_handoff,
        output_dir=args.output_dir,
        config=_build_config(args),
    )
    summary = {
        "status": "COMPLETE",
        "stage3_version": state["stage3_version"],
        "selected_frame": state["replay_context"]["selected_frame"],
        "candidate_tracks": state["metrics"]["candidate_tracks"],
        "pose_coverage_t0": state["metrics"]["PoseCoverageAtT0_given_stage2_candidate"],
        "accepted_pose_coverage_t0": state["metrics"]["AcceptedPoseCoverageAtT0_given_stage2_candidate"],
        "valid_pose_coverage_t0": state["metrics"]["ValidPoseCoverageAtT0_given_stage2_candidate"],
        "foot_pose_coverage_t0": state["metrics"]["FootPoseCoverageAtT0_given_stage2_candidate"],
        "artifacts": state["artifacts"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
