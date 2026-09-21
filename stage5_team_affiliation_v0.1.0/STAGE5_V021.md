# Stage5 v0.2.1 — fail-closed residual role recovery

v0.2.1 corrects two structural problems found in the five-sequence v0.2.0 smoke.
It does not tune thresholds on VALID and does not change legacy V0.

## Corrected GK gate

For each goal side, a residual becomes a goalkeeper candidate only when it has
enough metric observations and its median distance to that goal line is within
`goal_distance_m`. Pairwise temporal ordering is evaluated only between those
qualified candidates. An ordinary appearance outlier on the same half is not a
goalkeeper rival and cannot veto the deepest candidate.

Pairwise ordering among qualified goal candidates is recorded, but it is
diagnostic-only by default. The smoke artifact showed that a true referee can be
temporarily deeper than the goalkeeper in overlapping frames even while its
sequence-level median remains about ten metres farther from the goal. Therefore
`pairwise_ordering_required=false` unless a frozen GSR TRAIN calibration explicitly
shows that hard rejection improves the target metrics. A clear tie in median goal
depth still abstains.

Each residual now reports `role_decision_reason`, and the selected candidate adds
`role_gate_diagnostics` with candidate, compared, ignored, insufficient-overlap,
and failed-ordering track IDs.

## Abstention policy

The v0.2.0 referee rule produced false positives on ordinary players. Therefore
`referee_recovery_enabled=false` by default: a non-GK residual remains
`unknown_residual`. The old heuristic can only be enabled by an explicit frozen
configuration produced on GSR TRAIN.

Likewise, `defensive_tail_assignment_enabled=false` by default. The defensive-tail
calculation is retained under `goalkeeper_assignment`, including a possible
`suggested_team_id`, but the public prediction stays `team_id=null`,
`team_status=UNKNOWN`, and `assignment_method=DEFENSIVE_TAIL_DIAGNOSTIC_ONLY`.

These defaults trade coverage for correctness. They must not be switched on by
looking at VALID results.

## Verification commands

These are short tests, not the image benchmark:

```powershell
cd D:\NCKH\stage5_team_affiliation_v0.1.0
python -m pip install -e .
python -m pytest -q
python validate_stage5_v021.py
```

Then run a new five-sequence smoke in a new directory:

```powershell
$env:OMP_NUM_THREADS = "1"
python -u benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" --split valid --method residual-v3 --limit 5 `
  --output-dir ".\benchmark_results\gsr_v021_v3_5seq" --progress-every 1
```

Do not run the full VALID split yet. First inspect GK role recall, player-to-GK
contamination, outfield accuracy, and the reason-code distribution. Calibration
and any opt-in enabling must use TRAIN only; only after freezing that config is
VALID a fair held-out report.
