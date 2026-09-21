from __future__ import annotations

import argparse
import json

from stage4_metric3d.uncertainty_postprocess import apply_uncertainty_postprocess


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add camera-fixed Stage-3 pixel sensitivity to a frozen Stage-4 state without rerunning the base optimizer"
    )
    parser.add_argument("--base-state", required=True, help="Frozen metric_pose_3d_state JSON, preferably v0.1.6")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", choices=["selected_frame", "full_window"], default="selected_frame")
    parser.add_argument("--uncertainty-max-nfev", type=int, default=180)
    parser.add_argument("--stage3-state", default=None, help="Override the Stage-3 path stored in the frozen state")
    parser.add_argument("--camera-dir", default=None, help="Override the Stage-1 camera directory stored in the frozen state")
    parser.add_argument("--initializer-cache", default=None, help="Override the initializer cache stored in the frozen state")
    parser.add_argument("--track-id", action="append", dest="track_ids", help="Process only this track; repeat for multiple tracks")
    args = parser.parse_args()

    state = apply_uncertainty_postprocess(
        base_state=args.base_state,
        output_dir=args.output_dir,
        samples=args.samples,
        seed=args.seed,
        mode=args.mode,
        uncertainty_max_nfev=args.uncertainty_max_nfev,
        stage3_state=args.stage3_state,
        camera_dir=args.camera_dir,
        initializer_cache=args.initializer_cache,
        track_ids=args.track_ids,
    )
    print(json.dumps({
        "metric_pose_3d_state": state.get("artifacts", {}).get("metric_pose_3d_state"),
        "stage5_handoff": state.get("artifacts", {}).get("stage5_handoff"),
        "uncertainty_analysis": state.get("uncertainty_analysis"),
        "acceptance_gate": state.get("acceptance_gate"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
