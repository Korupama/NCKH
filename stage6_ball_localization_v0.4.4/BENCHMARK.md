# Stage 6 Evaluation Protocol — v0.5.0

Contact/fusion synthetic regression: `python validate_stage6_v050.py`.
Full regression: `python -m pytest tests -q` (use a fresh writable --basetemp if required).
`evaluation/contact_metrics.py` reports annotated contact track/region accuracy,
geometry coverage, BLE-X MAE/median/P90/P95. It reports null when GT is unavailable.
Fusion candidate count/dedup diagnostics are not recall. Use the existing 2D evaluator
with optimized GT for detector-inclusive recall/AP comparisons. No fusion accuracy
claim follows from reusing a YOLO-only cache. Anchor perturbations +/-0.25 and +/-0.50 m
produce deterministic X sensitivity, not calibrated uncertainty.

Separate gates: implementation tests, detector fusion accuracy, contact accuracy,
longitudinal accuracy, research freeze. The real frame86 smoke has no contact/3D GT
and therefore cannot pass the latter accuracy gates.

## Preserved v0.3 benchmarks

The following remain valid and are preserved unchanged in purpose:

- `run-2d`: detector benchmark.
- `reevaluate-2d`: cached original-vs-optimized GT comparison.
- `run-3d-oracle`: single-frame size-prior geometry ceiling diagnostic.
- `run-3d-e2e`: cached detector → single-frame 3D diagnostic.

All new reports identify the actually imported package via `runtime_provenance`.

## Temporal validation rule

Do **not** run the v0.4 temporal optimizer across the 810 SoccerNet-v3D test CSV rows in CSV order. Those rows are action/replay multi-view observations and are not a consecutive broadcast-frame trajectory.

A valid temporal benchmark requires a sequence where adjacent samples correspond to adjacent or time-indexed frames of the same continuous replay shot, together with ball XYZ or another defensible metric reference.

## Current v0.4 evidence levels

### Level A — deterministic synthetic regression

Used to verify implementation properties:

- noisy apparent diameter is refined;
- injected diameter spikes are down-weighted;
- short missing spans are interpolated;
- long missing spans remain missing;
- temporal world error decreases on a synthetic constant-velocity trajectory with known camera geometry.

This establishes implementation correctness under controlled conditions, not real-data accuracy.

### Level B — real replay consistency smoke

The previously generated 61-frame project replay is reprocessed using the real Stage-1 camera states and existing detector selections. Reported quantities include temporal coverage, active bounds, grounded compatibility and trajectory smoothness. Because no reference ball XYZ is available, these are consistency diagnostics only.

### Level C — future real temporal accuracy benchmark

Required before freezing v0.4 temporal hyperparameters or making an accuracy claim.
