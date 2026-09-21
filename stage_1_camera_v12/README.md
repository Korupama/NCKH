# Stage 1 Camera Calibration v12

> Local handoff: [dataset locations and download policy](../docs_ban_giao/DATASET_PATHS.md). Check existing datasets outside `D:\NCKH` before downloading; `D:\GSR` and `D:\SoccerNet` are existing roots, not folders to recreate inside the project.

From-scratch Stage-1 implementation for metric football-pitch camera calibration.

Package version: `0.12.0`. `CameraState` remains schema `1.2` until the held-out test and real temporal run are complete.

## What changed in v12

V12 keeps the v11 PnLCalib inference, candidate diagnostics, rendering, temporal-rescue code and SoccerNet benchmark runner, but **freezes capability-specific single-frame thresholds from the completed SoccerNet Calibration-2023 `valid` benchmark**.

The central change is that a general camera status no longer carries the full responsibility for downstream readiness. Two explicit frozen policies are exported:

```python
GroundGeometryPolicyConfig
Vertical3DPolicyConfig
```

The vertical/offside-3D direct-frame policy is:

```text
selected mode             full
RANSAC                     0
reprojection error         <= 5 px
keypoints                  >= 8
image quadrants            >= 3
hull ratio                 >= 0.01   # safety floor
x-span ratio               >= 0.20   # safety floor
y-span ratio               >= 0.06   # safety floor
```

These are **camera-calibration gates**, not a claim of offside-decision accuracy.

## Validation provenance

The policy was frozen from the completed `valid` report containing 3,212 frames. Reclassification of the v11 per-frame CSV gives:

```text
old vertical-ready         1115 / 3212 = 34.71%
v12 vertical-ready         1007 / 3212 = 31.35%
demoted from old gate      108
promoted                   0
mean frame calibration accuracy on v12-ready subset  ~= 0.9700
P10 frame calibration accuracy                       ~= 0.8889
fraction with frame calibration accuracy >= 0.90     ~= 0.8828
```

The exact audit is saved as `VALIDATION_POLICY_FREEZE_v12.json`.

## Ground vs vertical geometry

A low-error fallback can legitimately yield:

```text
CameraState.status         DEGRADED
ground_geometry.status     VALID
vertical_3d.status         DEGRADED
offside_3d_ready           False
```

This remains intentional. Pitch-plane accuracy does not by itself validate airborne body-point geometry.

## SoccerNet benchmark defaults

The runner keeps the PnLCalib SN23 settings:

```text
model size                 960 x 540
weights                    MV_kp / MV_lines
kp threshold               0.0712
line threshold             0.2571
max official reproj error  38 px
refine                     false
refine_lines               true
modes                      full, ground_plane, main
RANSAC                     0, 5, 10, 15, 25, 50
```

V12 benchmark reports use schema `1.1` and include capability-policy provenance plus per-frame vertical-policy reasons.

## Notebook workflow

1. Run Cell 0 (`%pip` only; no subprocess installer).
2. Restart the kernel once if binary packages changed.
3. Verify package version `0.12.0`.
4. Point to/download PnLCalib weights and SoccerNet Calibration-2023.
5. Optional single-frame QA.
6. Optional smoke/full-valid reruns. They are disabled by default because validation is already complete.
7. Cell 12C audits the frozen policy directly from an existing `valid_full_frames.csv` without neural inference.
8. Thresholds are now frozen; set only `RUN_FULL_TEST=True` for the held-out test run.
9. Do **not** retune the policy after viewing test results.
10. After test, run a real same-shot temporal CameraTimeline/rescue experiment.

## Tests

From the package root:

```bash
PYTHONPATH=. python -m pytest -q
```

V12 regression suite: `28 passed` in the generation environment.
