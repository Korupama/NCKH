from __future__ import annotations

import argparse
import json

from stage4_metric3d.backends.fixed_height_v03 import run_fixed_height_v03
from stage4_metric3d.backends.field_converter import FieldConverterV04Config, preflight_v04, run_v04
from stage4_metric3d.backends.field_converter.model_bundle import resolve_bundle
from stage4_metric3d.backends.sam3d_pitch_refined import Sam3DPitchRefinedConfig, preflight_v05, run_v05


def _add_common_v04(p: argparse.ArgumentParser) -> None:
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--camera-dir", required=True)
    p.add_argument("--sam3d-cache", required=True)
    p.add_argument("--output-dir", default="runs/stage4_v04")
    p.add_argument("--field-converter-repo", default=None)
    p.add_argument("--field-converter-python", default=None)
    p.add_argument("--fc-config", required=True)
    p.add_argument("--fc-checkpoint", required=True)
    p.add_argument("--fc-stats", required=True, help="Exact upstream normalization_stats.npz; not included in the public release as of v0.5 patch")
    p.add_argument("--fc-pitch-points", required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--world-alignment", choices=("auto", "none", "rotate_x_180", "rotate_y_180", "rotate_z_180"), default="auto")
    p.add_argument("--sam3d-sign", type=int, choices=(-1, 1), default=1)
    p.add_argument("--sequence-name", default="stage4_sequence")
    p.add_argument("--target-fps", type=float, default=50.0)
    p.add_argument("--no-save-fc-intermediate", action="store_true")
    p.add_argument("--timeout", type=int, default=None)


def _cfg_v04(args) -> FieldConverterV04Config:
    return FieldConverterV04Config(
        sequence_name=args.sequence_name, target_fps=args.target_fps,
        world_alignment=args.world_alignment, sam3d_sign=args.sam3d_sign,
        field_converter_device=args.device, field_converter_batch_size=args.batch_size,
        save_field_converter_intermediate=not args.no_save_fc_intermediate,
    )


def _bundle_v04(args):
    return resolve_bundle(
        field_converter_repo=args.field_converter_repo, python_exe=args.field_converter_python,
        config_path=args.fc_config, checkpoint_path=args.fc_checkpoint,
        normalization_stats_path=args.fc_stats, pitch_points_path=args.fc_pitch_points,
    )


def _add_common_v05(p: argparse.ArgumentParser, default_output: str) -> None:
    p.add_argument("--stage3-state", required=True)
    p.add_argument("--camera-dir", required=True)
    p.add_argument("--sam3d-cache", required=True, help="Stage4 v0.5 native MHR70 cache from run_sam3d_body_worker.py")
    p.add_argument("--output-dir", default=default_output)
    p.add_argument("--window-radius", type=int, default=15)
    p.add_argument("--reprojection-sigma-px", type=float, default=8.0)
    p.add_argument("--max-ground-prior-distance-m", type=float, default=2.5, help="Deprecated compatibility parameter; SAM disagreement no longer vetoes ground geometry")
    p.add_argument("--max-ground-refinement-m", type=float, default=0.75, help="Maximum distance from usable ground consensus to refined translation")
    p.add_argument("--max-translation-correction-m", type=float, default=5.0)
    p.add_argument("--disable-temporal", action="store_true")
    p.add_argument("--selected-frame", type=int, default=None, help="Override Stage-3 selected_frame for an exact zero-based frame index")
    p.add_argument("--preflight-only", action="store_true")


def _cfg_v05(args) -> Sam3DPitchRefinedConfig:
    return Sam3DPitchRefinedConfig(
        window_radius_frames=int(args.window_radius),
        reprojection_sigma_px=float(args.reprojection_sigma_px),
        max_ground_prior_distance_m=float(args.max_ground_prior_distance_m),
        max_ground_refinement_m=float(args.max_ground_refinement_m),
        max_translation_correction_m=float(args.max_translation_correction_m),
        use_temporal=not bool(args.disable_temporal),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 world-grounded player-pose pipeline")
    sub = parser.add_subparsers(dest="backend", required=True)

    refined = sub.add_parser("sam3d-pitch-refined", help="v0.5 production candidate: SAM3D metric pose + Stage1 pitch-constrained root-only refinement")
    _add_common_v05(refined, "runs/stage4_v05")

    direct = sub.add_parser("sam3d-direct", help="v0.5 ablation: use SAM3D pred_cam_t directly without root refinement")
    _add_common_v05(direct, "runs/stage4_v05_sam3d_direct")

    fc = sub.add_parser("field-converter-tcn", help="v0.4 optional backend; requires exact upstream normalization_stats.npz")
    _add_common_v04(fc); fc.add_argument("--preflight-only", action="store_true")

    fixed = sub.add_parser("fixed-height-v03", help="Retained v0.3 fixed-height baseline")
    fixed.add_argument("--stage3-state", required=True); fixed.add_argument("--camera-dir", required=True)
    fixed.add_argument("--output-dir", default="runs/stage4_v03_baseline")
    fixed.add_argument("--reference-height", type=float, default=1.80); fixed.add_argument("--selected-frame-only", action="store_true")

    args = parser.parse_args()
    if args.backend == "fixed-height-v03":
        result = run_fixed_height_v03(stage3_state=args.stage3_state, camera_dir=args.camera_dir, output_dir=args.output_dir, reference_height_m=args.reference_height, selected_frame_only=args.selected_frame_only)
        print(json.dumps({"backend": args.backend, "artifact": result.get("artifact")}, indent=2)); return 0

    if args.backend in {"sam3d-pitch-refined", "sam3d-direct"}:
        cfg = _cfg_v05(args)
        if args.preflight_only:
            result = preflight_v05(stage3_state=args.stage3_state, camera_dir=args.camera_dir, sam3d_cache=args.sam3d_cache, config=cfg, selected_frame=args.selected_frame)
            print(json.dumps(result, indent=2, ensure_ascii=False)); return 0 if result["ready"] else 2
        state = run_v05(stage3_state=args.stage3_state, camera_dir=args.camera_dir, sam3d_cache=args.sam3d_cache, output_dir=args.output_dir, config=cfg, refine=args.backend == "sam3d-pitch-refined", selected_frame=args.selected_frame)
        print(json.dumps({"backend": args.backend, "world_grounded_pose_state": state["artifacts"].get("world_grounded_pose_state"), "stage4_downstream_handoff": state["artifacts"].get("stage4_downstream_handoff"), "quality_gates": state.get("quality_gates")}, indent=2)); return 0

    cfg = _cfg_v04(args); bundle = _bundle_v04(args)
    if args.preflight_only:
        result = preflight_v04(stage3_state=args.stage3_state, camera_dir=args.camera_dir, sam3d_cache=args.sam3d_cache, bundle=bundle, config=cfg, probe_external_python=True)
        print(json.dumps(result, indent=2, ensure_ascii=False)); return 0 if result["ready"] else 2
    state = run_v04(stage3_state=args.stage3_state, camera_dir=args.camera_dir, sam3d_cache=args.sam3d_cache, bundle=bundle, output_dir=args.output_dir, config=cfg, timeout_s=args.timeout)
    print(json.dumps({"backend": args.backend, "world_grounded_pose_state": state.get("artifacts", {}).get("world_grounded_pose_state"), "stage4_downstream_handoff": state.get("artifacts", {}).get("stage4_downstream_handoff"), "quality_gates": state.get("quality_gates"), "camera_domain": (state.get("camera_domain") or {}).get("status")}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
