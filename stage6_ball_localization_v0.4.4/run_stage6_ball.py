from __future__ import annotations

import argparse
import json
from pathlib import Path

from ball_localization.geometry import HybridGeometryConfig, TemporalRefinementConfig
from ball_localization.pipeline import run_stage6


def _temporal_config(args) -> TemporalRefinementConfig:
    return TemporalRefinementConfig(
        diameter_second_diff_weight=args.temporal_diameter_second_diff_weight,
        diameter_first_diff_weight=args.temporal_diameter_first_diff_weight,
        max_gap_frames=args.temporal_max_gap_frames,
        range_sigma_fraction=args.temporal_range_sigma_fraction,
        position_second_diff_sigma_m=args.temporal_position_second_diff_sigma_m,
        position_third_diff_sigma_m=args.temporal_position_third_diff_sigma_m,
        pitch_margin_m=args.temporal_pitch_margin_m,
        max_height_m=args.temporal_max_height_m,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 6 ball detection/tracking/contact localization v0.5.1")
    p.add_argument("--stage1-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", choices=["yolo", "stage2-sst", "fusion", "yolo-first"], default="yolo")
    p.add_argument("--stage3-state")
    p.add_argument("--stage4-handoff")
    p.add_argument("--weights")
    p.add_argument("--stage2-entity-tracks")
    p.add_argument("--video")
    p.add_argument("--conf-floor", type=float, default=0.05)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=1920)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tracker", choices=["viterbi", "top1"], default="viterbi")
    p.add_argument(
        "--localization-mode",
        choices=["ground-first", "ground-only", "size-prior", "temporal-3d", "hybrid-3d", "contact-aware"],
        default="hybrid-3d",
    )
    p.add_argument("--ball-radius-m", type=float, default=0.11)
    p.add_argument("--pitch-margin-m", type=float, default=12.0)
    p.add_argument("--pitch-far-prior", type=float, default=0.20)
    p.add_argument("--progress-every", type=int, default=10)

    # v0.4 temporal geometry controls. Defaults are intentionally conservative
    # and remain visible in ball_trajectory_state.json for reproducibility.
    p.add_argument("--temporal-diameter-second-diff-weight", type=float, default=8.0)
    p.add_argument("--temporal-diameter-first-diff-weight", type=float, default=0.10)
    p.add_argument("--temporal-max-gap-frames", type=int, default=3)
    p.add_argument("--temporal-range-sigma-fraction", type=float, default=0.08)
    p.add_argument("--temporal-position-second-diff-sigma-m", type=float, default=0.75)
    p.add_argument("--temporal-position-third-diff-sigma-m", type=float, default=0.50)
    p.add_argument("--temporal-pitch-margin-m", type=float, default=6.0)
    p.add_argument("--temporal-max-height-m", type=float, default=30.0)
    p.add_argument("--hybrid-config-json", help="Frozen hybrid_calibration.json produced from ISSIA cameras 3–6")

    a = p.parse_args()
    if a.localization_mode == 'contact-aware' and not a.stage3_state:
        p.error('contact-aware requires --stage3-state; --stage4-handoff is required only for vertical contact')
    state = run_stage6(
        stage1_root=a.stage1_root,
        output_dir=str(Path(a.output_dir)/'hybrid_baseline') if a.localization_mode=='contact-aware' else a.output_dir,
        provider_kind=a.provider,
        weights=a.weights,
        stage2_entity_tracks=a.stage2_entity_tracks,
        video_path=a.video,
        conf_floor=a.conf_floor,
        top_k=a.top_k,
        imgsz=a.imgsz,
        device=a.device,
        tracker_kind=a.tracker,
        localization_mode='hybrid-3d' if a.localization_mode=='contact-aware' else a.localization_mode,
        ball_radius_m=a.ball_radius_m,
        pitch_margin_m=a.pitch_margin_m,
        pitch_far_prior=a.pitch_far_prior,
        progress_every=a.progress_every,
        temporal_config=_temporal_config(a) if a.localization_mode in {"temporal-3d", "hybrid-3d", "contact-aware"} else None,
        hybrid_config=HybridGeometryConfig.from_mapping((json.loads(Path(a.hybrid_config_json).read_text(encoding="utf-8")).get("best") or {}).get("config")) if a.hybrid_config_json else None,
    )
    if a.localization_mode=='contact-aware':
        from ball_localization.contact import refine_contact_file
        result=refine_contact_file(Path(a.output_dir)/'hybrid_baseline'/'ball_trajectory_state.json',a.stage1_root,a.stage3_state,a.stage4_handoff,a.output_dir)
        print(json.dumps(result['selected_frame_ball'],indent=2))
        return
    print(json.dumps({
        "status": state.status,
        "stage6_version": state.stage6_version,
        "selected_frame_ball": state.selected_frame_ball,
        "diagnostics": state.diagnostics,
        "artifacts": state.artifacts,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
