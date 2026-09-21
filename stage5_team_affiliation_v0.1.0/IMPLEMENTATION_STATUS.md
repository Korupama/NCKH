# Implementation status — Stage 5 v0.2.1

## v0.2.1 smoke-correction patch (2026-09-20)

- Pairwise GK ordering now compares only independently qualified goal candidates.
- Same-half appearance outliers no longer veto a goalkeeper.
- Referee recovery defaults to UNKNOWN until calibrated on GSR TRAIN.
- Defensive-tail GK team output is diagnostic-only until explicitly enabled by a frozen TRAIN config.
- Per-track reason codes and gate diagnostics are emitted.
- Package metadata: `0.2.1`; legacy V0 remains the default.
- Research accuracy frozen: false.

See `STAGE5_V021.md` and `PATCH_SUMMARY_v0.2.1.md`.

## v0.2.0 opt-in extension (2026-09-20)

- Package metadata: `0.2.0`.
- Legacy V0 remains the default and existing artifacts are preserved.
- Dominant-team/residual role-recovery code: implemented.
- Oracle `bbox_pitch` GSR benchmark protocol V1/V2/V3: implemented.
- Strict Stage1 replay cache seam: implemented; cache generation not implemented/run.
- Full regression: `44 passed`.
- Synthetic v0.2 validator: `PASS` (`31 passed`).
- Replay pipeline: NOT RUN for v0.2.
- SoccerNet-GSR v0.2 smoke/full benchmark: NOT RUN.
- Research accuracy frozen: false.

See `STAGE5_V020.md`. The older patch summary is archived in `../docs_ban_giao/tai_lieu_lich_su_20260920.zip`.

## Historical v0.1.1 status

Stage-5 team-affiliation algorithm: **v0.1.0 semantics unchanged**.
Package/tooling version: **0.1.1**.

Implemented and tested:

- team-affiliation replay pipeline;
- SoccerNet-GSR v1.3 benchmark reader;
- B0 bbox-colour intrinsic benchmark;
- B1 pose-guided benchmark with explicit oracle-track pose cache;
- external/official prediction import;
- track-level metrics and sequence bootstrap aggregation.

Validation status:

- automated tests: see QA report;
- synthetic benchmark validator: PASS;
- supplied real `Labels-GameState.json` schema smoke: PASS (SNGS-060, v1.3, 750 frames, 22 team-bearing tracks), but its image files are not mounted in this runtime, so **no real GSR accuracy is claimed here**;
- research accuracy remains NOT FROZEN until the user's local SoccerNet-GSR frames are benchmarked.
