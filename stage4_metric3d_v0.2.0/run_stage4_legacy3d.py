from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage4_metric3d.processor import preflight_stage4, run_stage4
from stage4_metric3d.schemas import Stage4Config


def _config_from_args(args) -> Stage4Config:
    cfg = Stage4Config()
    cfg.window_radius_frames = args.window_radius
    cfg.uncertainty_samples = args.uncertainty_samples
    cfg.uncertainty_mode = args.uncertainty_mode
    cfg.uncertainty_optimizer_max_nfev = args.uncertainty_max_nfev
    cfg.require_initializer = args.require_initializer
    cfg.temporal_weight = args.temporal_weight
    cfg.depth_prior_weight = args.depth_prior_weight
    cfg.ground_weight = args.ground_weight
    cfg.optimizer_max_nfev = args.max_nfev
    cfg.sparse_jacobian = not args.dense_jacobian
    cfg.optimizer_verbose = args.optimizer_verbose
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4 hybrid metric 3D player reconstruction")
    p.add_argument("--stage3-state", required=True, help="tracked_pose_2d_state.json")
    p.add_argument("--camera-dir", required=True, help="Stage-1 CameraState JSON directory")
    p.add_argument("--initializer-cache", default=None, help="RTMW3D initializer JSONL; omit for geometry-only fallback")
    p.add_argument("--output-dir", default="runs/stage4_v01")
    p.add_argument("--window-radius", type=int, default=13)
    p.add_argument("--uncertainty-samples", type=int, default=0)
    p.add_argument("--uncertainty-mode", choices=["selected_frame", "full_window"], default="selected_frame")
    p.add_argument("--uncertainty-max-nfev", type=int, default=180)
    p.add_argument("--temporal-weight", type=float, default=1.0)
    p.add_argument("--depth-prior-weight", type=float, default=1.0)
    p.add_argument("--ground-weight", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=250)
    p.add_argument("--dense-jacobian", action="store_true", help="Use dense numerical Jacobian for regression/debug")
    p.add_argument("--optimizer-verbose", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--require-initializer", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--allow-preflight-failure", action="store_true", help="Debug only; do not use for production")
    args = p.parse_args()

    cfg = _config_from_args(args)
    preflight = preflight_stage4(
        stage3_state=args.stage3_state,
        camera_dir=args.camera_dir,
        initializer_cache=args.initializer_cache,
        config=cfg,
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight["ready"] else 2

    state = run_stage4(
        stage3_state=args.stage3_state,
        camera_dir=args.camera_dir,
        initializer_cache=args.initializer_cache,
        output_dir=args.output_dir,
        config=cfg,
        strict_preflight=not args.allow_preflight_failure,
    )
    summary = {
        "metric_pose_3d_state": state.get("artifacts", {}).get("metric_pose_3d_state"),
        "stage5_handoff": state.get("artifacts", {}).get("stage5_handoff"),
        "selected_frame": state.get("replay_context", {}).get("selected_frame"),
        "metrics": state.get("metrics"),
        "acceptance_gate": state.get("acceptance_gate"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
