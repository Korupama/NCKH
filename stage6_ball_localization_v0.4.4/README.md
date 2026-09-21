# Stage 6 — Ball Detection, Tracking, Contact Association & Metric Localization v0.5.1

Read [v0.5.1 contact geometry hotfix](V051_PATCH_NOTES.md) first: FOOT no longer
requires a Stage4 anchor; non-FOOT has separate camera/anchor gates. It supersedes
the shared-gate instructions below. Runtime version is 0.5.1, while `pyproject.toml`
still reports 0.4.4; reconcile before distributing an installable release.
See the [workspace handoff guide](../README_BAN_GIAO.md).

## Contact-aware patch (2026-09-18)

The installation folder remains `stage6_ball_localization_v0.4.4`. Internal version is 0.5.0.
`hybrid-3d` stays the default. `--localization-mode contact-aware` is opt-in and requires
`--stage3-state` and `--stage4-handoff`. `--provider fusion` additionally requires
`--stage2-entity-tracks`; YOLO optimized box geometry is preserved when duplicates merge.
No new pretrained model or SDK is required: YOLO-opt/Ultralytics remains the detector;
contact association uses existing Stage3 poses, NumPy and OpenCV, not a learned classifier.

For a cached hybrid state, use `refine_ball_contact.py --state-json ... --stage1-root ...
--stage3-state ... --stage4-handoff ... --output-dir ...` (one command line).
Only t0 is refined; other frames and the original cache are preserved. See
`V050_PATCH_NOTES.md` for gates, limitations and real-run findings.

The remainder below documents the preserved v0.4 geometry implementation.

Stage 6 v0.4 adds **temporal ball geometry** to the existing offline replay pipeline. Stage 1 still owns camera geometry; Stage 2 still owns human entities. Stage 6 owns ball candidate selection and metric ball localization.

## v0.4 production path

```text
YOLO top-K candidates
        ↓
offline Viterbi ball path
        ↓
robust log-diameter refinement
        ↓
short-gap centre/diameter interpolation
        ↓
per-frame camera ray + size-prior range
        ↓
world-space temporal ray-range optimization
        ↓
BallState(t0) + temporal diagnostics
```

The optimizer does **not** move a directly observed ball centre off its measured image ray. It refines the apparent-size/range estimate while enforcing temporal and physical consistency.

## Why this was implemented

The v0.3 benchmarks showed that optimized 2D GT substantially changes the detector assessment and that candidate ranking is not the dominant 3D error source. The remaining error is strongly associated with monocular apparent-size/depth sensitivity. v0.4 therefore treats single-frame size-prior localization as an initializer rather than the final trajectory estimate.

## Quick reuse of an existing Stage-6 run

No YOLO rerun is required:

```powershell
python refine_ball_trajectory.py `
  --state-json ".\ball_trajectory_state.json" `
  --stage1-root "D:\NCKH\stage_1_camera_v12" `
  --output-dir ".\outputs\stage6_v040_temporal"
```

## Full production run

```powershell
python run_stage6_ball.py `
  --stage1-root "D:\NCKH\stage_1_camera_v12" `
  --output-dir ".\outputs\stage6_v040" `
  --provider yolo `
  --weights ".\weights\yolo-sn-ball-opt.pt" `
  --tracker viterbi `
  --localization-mode temporal-3d `
  --device cpu
```

Use `--device 0` for CUDA where supported.

## Main temporal diagnostics

Each refined frame records:

- `raw_diameter_px`
- `refined_diameter_px`
- `optimized_diameter_px`
- `raw_size_range_m`
- `optimized_range_m`
- `range_bounds_m`
- `ground_compatible`
- `ground_range_m`
- `center_imputed` / `diameter_imputed`
- `size_residual_sigma`
- `xyz_world_m`

The trajectory-level block reports number of temporal frames, imputed frames, active physical bounds, ground-compatible frames, temporal segments, and robust size-residual statistics.

## Dataset benchmark status

The existing SoccerNet-v3D 810-row test split remains useful for independent 2D and single-frame 3D evaluation. It is **not** a temporal sequence benchmark: action/replay rows represent synchronized multi-view observations rather than consecutive video frames. v0.4 therefore does not pretend to evaluate temporal accuracy by chaining those rows.

Temporal v0.4 accuracy should be evaluated on genuinely consecutive replay frames with 3D or sufficiently strong reference annotations. Until that dataset exists, the included real-replay run is a consistency smoke test only.

## Version provenance

Before a long benchmark, verify the imported package:

```powershell
python benchmark_ball.py provenance
```

Expected package version: `0.4.0`.

See `QUICKSTART.md`, `STAGE6_SPEC.md`, `BENCHMARK.md`, `IMPLEMENTATION_STATUS.md`, and the current `V051_PATCH_NOTES.md`.
