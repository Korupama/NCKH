# Local workspace deployment

No Git operation is required.

Extract this archive so the folder is:

```text
D:\NCKH\stage9_offside_position_v0.1.0
```

Then:

```powershell
cd D:\NCKH\stage9_offside_position_v0.1.0
pip install -r requirements.txt
python -m pytest -q
python benchmark_stage9.py oracle --output-dir ".\outputs\stage9_oracle"
python web_demo.py --demo
```

Browse to `http://127.0.0.1:8099`.
