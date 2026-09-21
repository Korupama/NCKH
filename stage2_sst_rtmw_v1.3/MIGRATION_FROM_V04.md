# Migration from `sst_rtmw_3d_world_project_v04`

The previous v0.4 artifact mixed perception, tracking, camera, 3D lifting, world placement and offside logic. This package extracts only the Stage-2 responsibilities.

## Retained

- SST football-specific detector;
- RTMW WholeBody inference engine;
- decision-frame-anchored bidirectional tracking concept;
- exact global source-frame provenance;
- conservative failure semantics.

## Changed

1. **Stage 1 owns the replay window.** The old v0.4 context selector is no longer the temporal source of truth.
2. **Cross-class physical-human consolidation now runs before RTMW.** This fixes Player/Goalkeeper/Referee duplicate hypotheses that class-local NMS cannot remove.
3. **Referee/staff are not deleted.** They are retained internally as excluded humans, hidden by default, and prevented from entering Stage 3.
4. **RTMW is a shared raw cache.** Stage 2 may use body points for association but does not claim Stage-3 pose quality.
5. **Tracks are re-keyed.** Downstream pose evidence is indexed by persistent `track_id` + global `frame_index`, not frame-local SST detection IDs.
6. **Old legal-body flags are removed from the Stage-2 public contract.** Legal body semantics belong downstream.
7. **Ball stays auxiliary.** SST ball detections are retained for later comparison but Stage 2 does not track/triangulate the ball.

## Removed from this Stage-2 package

- PnLCalib orchestration;
- KASportsFormer/FMPose3D lifting;
- world placement;
- team/attack inference;
- legal-body 3D proxies;
- second-last-opponent logic;
- offside decision logic.

Those remain responsibilities of later stages.

## Legacy migration mode

`legacy_adapter.py` can read the old `*_sst_pose.json` files to regression-test the new consolidation and tracking contracts. It is not equivalent to a fresh Stage-2 run because old RTMW inference happened before physical-human consolidation; this limitation is explicitly recorded in provenance.
