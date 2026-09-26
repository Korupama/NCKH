# Stage 8 — Offside Reference Geometry v0.1.0

Deterministic Stage 8 for the offline replay pipeline.

It consumes Stage 4, Stage 6 and Stage 7 and emits the second-last-opponent / ball reference plane. It does **not** classify any attacker as being in an offside position.

## Inputs

- Stage 4: `stage4_downstream_handoff.json` (`stage4-downstream-handoff-2.1`)
- Stage 6: `stage6_downstream_handoff.json` (`stage6-downstream-handoff-1.0`, `stage8` block)
- Stage 7: `game_state_context.json` with `status=VALID`, `attack_direction.s`, and complete `sets.opponents`

## Core equations

```text
q = sX
q_i = max legal-landmark q for opponent i
q_2nd = second largest opponent q
q_ball = goalward endpoint of ball X extent
q_ref = max(q_2nd, q_ball)
X_ref = s*q_ref
```

## Install/test

From this folder:

```powershell
python -m pytest -q
```

Expected for the packaged v0.1.0 implementation: all tests pass.

## Oracle benchmark

```powershell
python benchmark_stage8.py oracle --output-dir ".\outputs\stage8_oracle"
```

Expected:

```text
status = PASS
oracle_logic_accuracy = 1.0
```

## Example preflight

```powershell
python preflight_stage8.py --stage4 ".\examples\stage4.json" --stage6 ".\examples\stage6.json" --stage7 ".\examples\stage7.json"
```

## Example run

```powershell
python run_stage8.py --stage4 ".\examples\stage4.json" --stage6 ".\examples\stage6.json" --stage7 ".\examples\stage7.json" --output ".\outputs\example\offside_reference_state.json" --plot ".\outputs\example\stage8_longitudinal_qa.png"
```

The plot is optional and requires matplotlib. Core Stage 8 has no third-party runtime dependency.

## Current project frame 104

Once Stage 7 has been unblocked and has produced a `VALID` project context, run:

```powershell
python preflight_stage8.py `
  --stage4 "D:\NCKH\stage4_metric3d_v0.2.0\runs\stage4_v051_frame104\sam3d-pitch-refined\stage4_downstream_handoff.json" `
  --stage6 "D:\NCKH\stage6_ball_localization_v0.4.4\outputs\replay_t0_104_contact_v051\stage6_downstream_handoff.json" `
  --stage7 "D:\NCKH\stage7_game_state_v0.1.0\outputs\project_replay_104\game_state_context.json"
```

Then:

```powershell
python run_stage8.py `
  --stage4 "D:\NCKH\stage4_metric3d_v0.2.0\runs\stage4_v051_frame104\sam3d-pitch-refined\stage4_downstream_handoff.json" `
  --stage6 "D:\NCKH\stage6_ball_localization_v0.4.4\outputs\replay_t0_104_contact_v051\stage6_downstream_handoff.json" `
  --stage7 "D:\NCKH\stage7_game_state_v0.1.0\outputs\project_replay_104\game_state_context.json" `
  --output ".\outputs\project_replay_104\offside_reference_state.json" `
  --plot ".\outputs\project_replay_104\stage8_longitudinal_qa.png"
```

If Stage 7 is still `UNRESOLVED` because of the known referee-role issue, Stage 8 will correctly fail closed with `STAGE7_CONTEXT_UNRESOLVED`.

## Boundary to Stage 9

Stage 8 does not inspect attacker body extents for a verdict. Stage 9 should consume this reference and then apply attacker/toucher/own-half logic to output `OFFSIDE_POSITION` vs `NOT_OFFSIDE_POSITION` only.
