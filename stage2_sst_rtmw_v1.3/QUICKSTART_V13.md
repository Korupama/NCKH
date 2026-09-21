# Quick start — Stage 2 v1.3 M1

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

## 1. Install and regression-test

```powershell
cd D:\NCKH\stage2_sst_rtmw_v1.3
python -m pip install -e ".[benchmark]"
pytest -q
```

Expected package regression result for this artifact: `16 passed, 1 skipped` in the packaging runtime. The skip is an optional Stage-1 mounted-workspace integration test.

## 2. Fresh M1 quick benchmark

v1.3 needs one fresh perception run because v1.2 caches do not contain low-score SST output.

```powershell
python benchmark_stage2.py run --soccernet-root "D:\GSR" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --sst-device cuda --rtmw-device cpu --raw-human-score-floor 0.20 --output-dir ".\benchmark_results\quick_v13" --progress-every 10
```

Resume after interruption by running the exact same command again.

Check status without loading models:

```powershell
python benchmark_stage2.py status --output-dir ".\benchmark_results\quick_v13"
```

## 3. Inspect the M1 diagnostics

Primary summary fields:

- `CandidatePrecision`, `CandidateRecall`
- `RefereeLeakageRate`
- `AnyDuplicateRate_before/after`
- `CrossClassConflictRate_before/after`
- `HOTA`, `AssA`, `IDF1`
- `AnchorCoverage`, `ConditionalTCR`, `TCR`

## 4. Sweep Player/GK thresholds without rerunning models

```powershell
python benchmark_stage2.py sweep-thresholds --soccernet-root "D:\GSR" --split valid --benchmark-dir ".\benchmark_results\quick_v13" --player-thresholds "0.30,0.35,0.40,0.45,0.50" --goalkeeper-thresholds "0.30,0.35,0.40,0.45,0.50"
```

Outputs:

```text
benchmark_results\quick_v13\threshold_sweep\
  threshold_sweep.json
  threshold_sweep.csv
```

This sweep evaluates selected-frame detection/role only. Once a threshold pair is selected, run a new quick benchmark with those active thresholds so RTMW/tracking sees the promoted candidates.

Example:

```powershell
python benchmark_stage2.py run --soccernet-root "D:\GSR" --split valid --protocol quick --sst-checkpoint ".\weights\model.pth" --rtmw-model ".\weights\rtmw_l_384x288.onnx" --player-threshold 0.40 --goalkeeper-threshold 0.40 --raw-human-score-floor 0.20 --output-dir ".\benchmark_results\quick_v13_p040_g040" --progress-every 10
```

## 5. Do not run full validation yet

Review the fresh M1 quick benchmark and threshold/rescue effects first. Full validation should start only after the quick configuration is frozen.
