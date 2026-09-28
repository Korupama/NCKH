from __future__ import annotations

import argparse, json
from pathlib import Path
from stage5_team_affiliation import Stage5Config, run_stage5


def main():
    ap = argparse.ArgumentParser(description="Stage 5 v0.2.2 team affiliation (legacy V0 and opt-in residual recovery)")
    ap.add_argument("--stage3-state", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--stage2-state", default=None)
    ap.add_argument("--stage4-handoff", default=None, help="Optional Stage4 world-pose handoff used for goalkeeper spatial affiliation")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["legacy-v0", "residual-v1", "residual-v2", "residual-v3"], default="legacy-v0")
    ap.add_argument("--pitch-state", help="Stage1 metric pitch-track cache; required for residual V2/V3")
    ap.add_argument("--residual-config", help="Frozen ResidualConfig JSON")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--min-torso-frames", type=int, default=3)
    ap.add_argument("--min-lower-frames", type=int, default=2)
    args = ap.parse_args()
    cfg = Stage5Config(
        sample_every_n_frames=args.sample_every,
        max_samples_per_track=args.max_samples,
        min_valid_torso_frames=args.min_torso_frames,
        min_valid_lower_body_frames=args.min_lower_frames,
    )
    if args.method != "legacy-v0":
        from stage5_team_affiliation.residual_replay import run_residual_replay
        from stage5_team_affiliation.residual_config import ResidualConfig
        residual_cfg = ResidualConfig(**json.loads(Path(args.residual_config).read_text(encoding="utf-8"))) if args.residual_config else ResidualConfig()
        state = run_residual_replay(stage3_state=args.stage3_state, stage2_state=args.stage2_state, video_path=args.video,
                                    output_dir=args.output_dir, pitch_state=args.pitch_state, variant=args.method[-2:].upper(),
                                    config=cfg, residual_config=residual_cfg)
        print(json.dumps({"stage5_version": state["stage5_version"], "artifacts": state["artifacts"]}, indent=2))
        return
    if args.pitch_state or args.residual_config:
        ap.error("Pitch/residual configuration only applies to residual methods")
    state = run_stage5(
        stage3_state=args.stage3_state,
        video_path=args.video,
        stage2_state=args.stage2_state,
        stage4_handoff=args.stage4_handoff,
        output_dir=args.output_dir,
        config=cfg,
    )
    print(json.dumps({
        "stage5_version": state["stage5_version"],
        "selected_frame": state["replay_context"]["selected_frame"],
        "metrics": state["metrics"],
        "artifacts": state["artifacts"],
    }, indent=2))

if __name__ == "__main__":
    main()
