# Stage 5 v0.2.1 — Team Affiliation & Residual Role Recovery

See [STAGE5_V021.md](STAGE5_V021.md) for the corrected opt-in dominant-teams/residual-role
algorithm, oracle-geometry benchmark, ablations, schemas, limitations and commands.
Legacy V0 remains the default. Historical v0.1 documentation follows below.

Stage 5 assigns each tracked human to a **team cluster** without resolving attacking/defending semantics.

## Scope

Owned by Stage 5:
- outfield team clustering (`TEAM_0`, `TEAM_1`);
- referee exclusion;
- conservative goalkeeper affiliation;
- track-level appearance aggregation;
- downstream track -> team handoff.

Not owned by Stage 5:
- attacking team;
- attack direction;
- second-last opponent;
- ball/toucher inference;
- legal-body geometry;
- offside decision.

Cluster IDs are arbitrary. Stage 7 determines the attacking team from the Stage 6 toucher track.

## Method

```text
Stage 3 WholeBody133 + replay frames
        |
        +-> pose-guided torso polygons
        |       -> HSV/Lab appearance features
        |       -> track-level robust median
        |       -> KMeans(k=2) on outfield players
        |
        +-> lower-body appearance
                -> conservative goalkeeper affinity

referee -> excluded from clustering
```

No pretrained model is used directly in Stage 5 v0.1. RTMW is an upstream Stage 3 model, not a Stage 5 dependency.

## Inputs

Required:
- `tracked_pose_2d_state.json` (`tracked-pose-2d-state-1.0`);
- original replay video in the same RAW_DISTORTED_PIXEL geometry.

Optional:
- Stage 2 `stage2_entity_tracks.json` for role/identity alignment checks.

Stage 4 is deliberately **not** a dependency.

## Outputs

- `team_affiliation_state.json` (`team-affiliation-state-1.0`)
- `stage5_downstream_handoff.json` (`stage5-downstream-handoff-1.0`)
- `selected_frame_team_affiliation.png`

## Run

```powershell
python run_stage5.py `
  --stage3-state "D:\NCKH\...\tracked_pose_2d_state.json" `
  --video "D:\NCKH\stage_1_camera_v12\download.mp4" `
  --output-dir ".\outputs\stage5_v010"
```

Optional Stage 2 consistency check:

```powershell
  --stage2-state "D:\NCKH\...\stage2_entity_tracks.json"
```

## Validation

```powershell
python -m pytest -q
python validate_stage5_v010.py
```

The release validation is synthetic/contract-only. It does **not** claim SoccerNet-GSR team accuracy.

## Third-party quantitative benchmark (v0.1.1 tooling)

Use `benchmark_soccernet_gsr.py` for the SoccerNet-GSR v1.3 track-level benchmark. The benchmark never uses the project replay/frame 104 for quantitative metrics. See `BENCHMARK.md` for B0/B1/external-baseline protocols and commands.
