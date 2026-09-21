# M1 improvement plan and experiment protocol

## Goal

Improve the Stage-2 front-end while preserving the existing SST detector, RTMW shared cue, and decision-frame-anchored tracking architecture. M1 targets the failure modes observed in the quick SoccerNet-GSR run before any detector fine-tuning.

## Failure modes targeted

1. **Player/GK cross-class duplicate on one physical person** — addressed by geometry-aware consolidation and footballer super-class semantics.
2. **Short detector dropouts during pan/motion blur** — addressed by low-score rescue observations and a larger temporal gap budget.
3. **Missing entity at t0 vs poor continuity after anchoring** — separated by `AnchorCoverage` and `ConditionalTCR`.
4. **Misleading duplicate evaluation** — fixed with identical before/after duplicate definitions.
5. **Threshold tuning cost** — low-score SST output is cached so Player/GK selected-frame sweeps do not rerun SST/RTMW.

## Two-threshold policy

For each human class, SST is retained to a low floor (default 0.20). The normal production thresholds remain independently configurable. A detection below the production threshold is never allowed to create an entity at the decision frame. After physical-human consolidation, an eligible low-score footballer hypothesis may only rescue an already-existing footballer track when the normal association pass failed.

This preserves high precision at `t0` while recovering temporal observations that were previously lost during blur/occlusion.

## Recommended M1 experiment order

1. Run one SoccerNet sequence with v1.3 and visually inspect normal/rescue counts.
2. Run the same 10-sequence quick protocol.
3. Run `sweep-thresholds` on the v1.3 cache. Start with Player/GK grid 0.30–0.50.
4. Select thresholds primarily by CandidateRecall and CandidatePrecision, subject to referee leakage and residual duplicate gates.
5. Re-run quick with the selected active thresholds so RTMW/tracking reflects the promoted normal candidates.
6. Only then sweep tracker parameters (`max_gap`, assignment cost, rescue assignment cost) using the new cache/evaluation checkpoints.
7. If crowded/small-player failures remain after M1, move to M2: hard-example SST fine-tuning and broadcast-specific augmentation.

## M2 is intentionally not implemented in this release

Training code/model fine-tuning is deferred until M1 establishes the residual error distribution. Candidate training directions are hard-example sampling, motion-blur/downscale/compression/occlusion augmentation, cost-sensitive footballer-vs-non-footballer objectives, and optionally a separate track-level RoleNet. RTMW fine-tuning belongs to Stage 3, not this Stage-2 M1 update.
