# Stage 8 Evaluation Protocol v1

## Tier A — deterministic oracle benchmark

Synthetic world-space fixtures test software correctness only.

Target: `oracle_logic_accuracy = 1.0`.

Required cases include:

- left-to-right and right-to-left attack
- ball ahead / second-last opponent ahead / level
- goalkeeper among last two opponents
- tied second-last coordinates
- arms excluded from legal-landmark extent
- head/shoulder/foot legal extrema
- frame mismatch
- missing opponent geometry
- fewer than two opponents
- unusable ball geometry

Any Tier-A failure is an implementation defect.

## Tier B — public proxy evidence

Public pitch-position/team datasets may be used to study opponent ordering and second-last-opponent identity, but ground-point/player-position labels are only a proxy for legal-body extent. Do not turn that proxy into a claim of exact legal-body X accuracy.

## Tier C — frozen project/expert benchmark

Use independently labeled replay cases. Partial GT is allowed. Recommended labels:

- selected frame
- attack direction
- opponent set
- second-last opponent identity (or tied candidate set)
- reference source: BALL / SECOND_LAST_OPPONENT / LEVEL
- optional reference X only when independently measured

Primary metrics:

- status accuracy
- second-last-opponent agreement
- reference-source agreement
- opponent-ranking exact match
- reference-X MAE when independent metric GT exists
- valid/abstention rate

Missing GT is `NOT_EVALUATED`; never substitute Stage-8 predictions as GT.

## Freeze discipline

Do not tune metric thresholds on held-out test cases. Numerical tie/comparison epsilon in v0.1.0 is machine-level (`1e-9 m`) and is not a football tolerance or uncertainty margin.
