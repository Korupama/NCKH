# Stage 7 Evaluation Protocol v2

## Purpose

Stage 7 is deterministic glue over Stage 1, Stage 5 and Stage 6. Its evaluation must separate software correctness from upstream accuracy.

### Tier A - deterministic oracle benchmark

Use synthetic/oracle inputs and require exactly 100% logic accuracy. Any failure is a software defect.

### Tier B - public-context evidence

Use public third-party datasets only for fields they actually label:

- SoccerNet-GSR: role, team side, track identity, pitch position. Useful for participant team/role and referee-context evaluation. It does not directly label the manual-t0 toucher or attacking possession team.
- FOOTPASS: event frame, event team, jersey/player identity, tracking/tactical context. Useful for actor/toucher-related and attacking-team-at-event evaluation when an identity mapping is available. It does not provide referee-set GT for Stage 7 and its `LEFT_TO_RIGHT` field is not treated as verified Stage-7 direction GT by this protocol.

Do not manufacture missing GT by reusing Stage-7 predictions or the centre-ray rule itself.

### Tier C - frozen project benchmark

Create an independently labeled manifest with partial labels allowed. Every case must state `provenance.independent_gt`.

The v2 evaluator reports a denominator for every metric. Missing labels are `NOT_EVALUATED`, never automatic failures or passes.

## Metrics

- `status_accuracy`
- `frame_alignment_accuracy`
- `toucher_track_accuracy`
- `attacking_team_accuracy`
- `attack_direction_accuracy`
- `attacker_set_exact_match`
- `opponent_set_exact_match`
- `referee_exclusion_exact_match`
- `participant_set_micro_precision/recall/f1`
- `referee_exclusion_precision/recall/f1`
- `full_game_state_exact_match`
- `invariant_pass_rate`
- `pred_valid_rate`, `abstention_rate`, `toucher_handoff_presence_rate`

Participant micro-F1 uses tagged labels (`A:<track>` and `O:<track>`), so placing the right track in the wrong side is counted as an error.

## Engineering profiles

The package contains two internal profiles:

- `pilot-minimum`: first executable end-to-end gate; minimum 30 cases for the main metrics.
- `research-target`: stronger target after pilot decisions are frozen; normally 100 independent cases.

These are internal engineering targets informed by related third-party results, not official SoccerNet or FOOTPASS thresholds.

Attack-direction targets are explicitly internal-only because no directly comparable public benchmark was found for the current Stage-7 semantics. In particular, the current `s = sign(X_view)` rule is a project scope assumption and must be evaluated against independent labels rather than declared correct by construction.

## Commands

```powershell
python benchmark_stage7_v2.py references --output-dir ".\outputs\stage7_references"
python benchmark_stage7_v2.py oracle --output-dir ".\outputs\stage7_oracle_v2"
python benchmark_stage7_v2.py manifest-template --output ".\data\stage7_eval_manifest_v2.json"
python benchmark_stage7_v2.py evaluate --manifest ".\data\stage7_eval_manifest_v2.json" --output-dir ".\outputs\stage7_eval_v2" --profile pilot-minimum
```

A profile returns `INCOMPLETE` when there are too few independently labeled cases. It never converts missing labels into a pass.
