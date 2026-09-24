# Stage 7 — Game-State Context v0.1.0

Stage 7 is a deterministic context module between Stage 1/5/6 and Stage 8.
It does **not** choose the second-last opponent and does **not** make the offside decision.

## Inputs

- Stage 1: `centre_ray_pitch_hit_m`
- Stage 5: active tracked humans with `track_id`, `team_id`, role/referee state
- Stage 6: selected-frame `contact.track_id`

## Outputs

`game_state_context.json` containing:

- toucher track/team
- attacking team
- attack direction `s ∈ {-1,+1}`
- attacker/opponent sets
- referees excluded
- diagnostics + invariant checks

Direction convention:

- `X_view > 0` → `s=+1` → LEFT_TO_RIGHT
- `X_view < 0` → `s=-1` → RIGHT_TO_LEFT

## Run

```powershell
python run_stage7.py --stage1 .\examples\stage1.json --stage5 .\examples\stage5.json --stage6 .\examples\stage6.json --output .\outputs\game_state_context.json
```

## Oracle logic benchmark

```powershell
python benchmark_stage7.py oracle --output-dir .\outputs\stage7_oracle
```

Expected target: `oracle_logic_accuracy = 1.0`.

## Frozen manifest benchmark

```powershell
python benchmark_stage7.py manifest-template --output .\data\stage7_eval_manifest.json
python benchmark_stage7.py evaluate --manifest .\data\stage7_eval_manifest.json --output-dir .\outputs\stage7_eval
```

Metrics:

- attacking-team accuracy
- attack-direction accuracy
- attacker/opponent exact-match
- referee-exclusion exact-match
- set micro-F1
- full GameState exact-match
- invariant pass rate

## Important boundary

Stage 7 intentionally does not rank opponents. The second-last opponent is Stage 8.
The toucher remains in `attackers`; Stage 9 excludes the toucher from offside candidates.
