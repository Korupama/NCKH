# Apply Stage 7 v0.1.0

This package is standalone. Copy the `stage7_game_state_v0.1.0` directory next to the other stage packages, or copy its contents into a new Stage 7 workspace.

Recommended project layout:

```text
NCKH/
├── stage6_ball_localization_v0.4.4/
└── stage7_game_state_v0.1.0/
```

Smoke test:

```powershell
cd D:\NCKH\stage7_game_state_v0.1.0
python benchmark_stage7.py oracle --output-dir .\outputs\stage7_oracle
python run_stage7.py --stage1 .\examples\stage1.json --stage5 .\examples\stage5.json --stage6 .\examples\stage6.json --output .\outputs\game_state_context.json
```
