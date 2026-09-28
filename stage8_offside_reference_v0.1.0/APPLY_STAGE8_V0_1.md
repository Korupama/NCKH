# Local use — no Git operations

1. Extract `stage8_offside_reference_v0.1.0.zip` under `D:\NCKH` so the result is:

```text
D:\NCKH\stage8_offside_reference_v0.1.0
```

2. Run:

```powershell
cd D:\NCKH\stage8_offside_reference_v0.1.0
python -m pytest -q
python benchmark_stage8.py oracle --output-dir ".\outputs\stage8_oracle"
```

3. No source workspace is overwritten and no Git command is required. Commit it yourself when ready.
