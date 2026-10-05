# Stage 4 Benchmark Protocol

## 1. Why multiple metrics are necessary

A monocular 3D model can have good root-aligned pose while being placed incorrectly on the pitch. Offside depends strongly on absolute longitudinal position, so Stage 4 reports both conventional 3D-pose metrics and pitch/world metrics.

## 2. Core metrics

### Local / articulation metrics

- `MPJPE`: mean Euclidean joint error in metric world coordinates;
- `RootAlignedMPJPE`: translation removed at hip root;
- `PA-MPJPE`: similarity-Procrustes aligned articulation error.

### Global metric metrics

- `GlobalMPJPE` without root alignment;
- per-anchor longitudinal error \(|\hat X-X|\);
- mean/median/P90 longitudinal error;
- ground/contact residual diagnostics.

### Offside-oriented metric

Before Stage 5 builds full body surfaces, Stage 4 reports **Goalward Anchor Longitudinal Error (GALE)**:

\[
GALE=\left|\max_{j\in A}s\hat X_j-\max_{j\in A}sX_j^{GT}\right|,
\]

where `A` is the project-scope candidate anchor set and \(s\in\{-1,+1\}\) is supplied only for evaluation. GALE is not the final Legal Body Longitudinal Error; that belongs to Stage 5 after surface construction.

### Pairwise ordering

For nearby players, report whether predicted longitudinal order matches GT. Recommended strata:

- GT separation < 2.0 m;
- < 1.0 m;
- < 0.5 m;
- < 0.25 m.

This catches offside-critical ranking failures that average MPJPE can hide.

## 3. Canonical GT JSON

`benchmark_stage4.py evaluate-canonical` accepts:

```json
{
  "schema_version": "metric-pose23-gt-1.0",
  "poses": [
    {
      "frame_index": 86,
      "track_id": "track_003",
      "xyz23_world_m": [[0,0,0], "... 23 x 3 ..."]
    }
  ]
}
```

Alternatively each record may contain a `joints` object keyed by Stage-4 `POSE23_NAMES`.

## 4. Independent Stage-4 model-only lane

The primary improvement/evaluation lane is independent of Stage 1 and Stage 3.
Each benchmark record must provide, or be convertible to, the following:

- RGB image or person crop;
- person bounding box and stable record/track identifier;
- camera intrinsics/extrinsics when global pitch/world metrics are claimed;
- metric 3D pose/position ground truth and visibility mask;
- dataset release, license, sequence split and coordinate convention.

Stage 4 uses its own pretrained model/backend and optimizer. Stage-1 camera
states and Stage-3 pose outputs are not used as labels and are not required for
this lane. A benchmark camera is not the same artifact as the project's Stage-1
camera timeline; both may be evaluated, but they must be named separately.

### Manifest and validator

The model-only intake contract is `stage4-model-only-benchmark-manifest-1.0`.
An example is in `STAGE4_MODEL_ONLY_MANIFEST.example.json`. Validate a real
manifest before inference:

```powershell
conda run -n nckh-env python -c "from stage4_metric3d.model_only import load_model_only_manifest; m=load_model_only_manifest(r'path/to/manifest.json'); print(m.sha256, len(m.records))"
```

The validator requires real dataset release/license metadata, sequence split,
RGB image, in-image bbox, benchmark camera, and one
`metric-pose23-gt-1.0` file per record. It resolves paths relative to the
manifest, checks image/camera dimensions and camera validity, validates Pose23
shape/visibility, and records SHA256 for every referenced input. It rejects
missing/empty/placeholder assets and any Stage-1/Stage-3 path in the manifest.

The contract is deliberately strict: H3WB or SportsPose records may be used
only after conversion to the declared coordinate scope. If a release does not
provide pitch-world metric camera/GT, use it for root-relative articulation
only through a separate contract; do not claim global football-pitch accuracy.

`evaluate_model_only.py` can evaluate a saved output independently of
inference. It revalidates the manifest/output provenance and emits common
cohort direct-vs-ground-first metrics plus failure attribution by model
coverage, native reprojection diagnostics, and player image scale. These are
diagnostics, not causal claims or calibrated confidence.

For Phase 3, use `STAGE4_MODEL_ONLY_SYSTEMS.example.json` with
`compare_model_only_systems.py`. All listed systems must consume the exact
same benchmark manifest and record cohort. The comparator checks checkpoint
hashes and output schemas before paired comparison; backend disagreement is a
diagnostic until held-out evidence supports promotion.

## 5. Recommended dataset lanes

### H3WB

Use to validate the relative WholeBody3D initializer. This is not a global football-camera benchmark.

### SportsPose

Use to evaluate dynamic sports articulation / temporal lifting. It is useful for comparing RTMW3D and sports-specific lifters but does not by itself validate pitch-world placement.

### WorldPose

Preferred project-domain benchmark because it provides soccer broadcast imagery, 3D pose and camera information. Data access/licensing is external to this package; Stage 4 deliberately does not fabricate an adapter against unseen local file layouts. Convert accessed data to the canonical GT JSON above, or add a dataset-specific adapter once the real release layout is available locally.

## 6. Required ablations

For a publication-grade experiment, compare:

1. geometry/body-prior only;
2. RTMW3D relative prior only where meaningful;
3. RTMW3D + calibrated camera + ground;
4. RTMW3D + calibrated camera + ground + temporal (production);
5. KASportsFormer + the same metric optimizer;
6. GT-2D oracle + real camera;
7. real Stage3 + GT-camera oracle where a benchmark provides camera GT.

### Optional project-integration attribution matrix

When the project integration claim is evaluated, use four named lanes with the
same tracks, initializer and optimizer configuration:

1. `GT2D_GTCAMERA`;
2. `STAGE3_GTCAMERA`;
3. `GT2D_STAGE1CAMERA`;
4. `STAGE3_STAGE1CAMERA`.

`benchmark_stage4.py evaluate-lanes` validates this optional integration matrix,
evaluates them against the same canonical GT and reports deltas from the
GT2D+GTCamera oracle. The manifest shape is provided in
`BENCHMARK_LANES.example.json`.

For the independent model-only claim, compare the same benchmark record cohort
across frozen Stage-4 pretrained backends and refinement configurations. No
Stage-1/3 lane is required.

### Uncertainty diagnostics

When selected-frame intervals are present, canonical evaluation additionally
reports empirical X/Y/Z coverage, interval width, legal-anchor X coverage and
min/max legal-X coverage. Coverage of the v0.2 camera-fixed pixel sensitivity
must not be described as calibrated confidence.

These lanes separate errors from 2D pose, 3D prior, camera calibration, contact inference and metric optimization.

## 7. Synthetic benchmark

`python benchmark_stage4.py synthetic-smoke` is a deterministic implementation check with exact generated GT. It validates geometry, optimizer, serialization, and the evaluator. It must never be reported as real model accuracy.

## 8. Required comparison hierarchy

Stage 4 uses three separate comparison layers.  They answer different
questions and must not be collapsed into one table without the coordinate-scope
labels.

### A. Pipeline regression against the frozen release

Use `compare-pipelines` for every Stage-4 release.  This comparison needs no GT
and reports paired XYZ drift, finite-joint loss/gain and selected-frame status
transitions.  For v0.2 uncertainty post-processing over the frozen v0.1.6 point
estimate, the expected gate is zero finite-joint loss and effectively zero XYZ
drift.  A status-only change is reported separately because a semantics repair
may intentionally change `REJECTED` to `MISSING` without changing coordinates.

This test detects software regressions; it does **not** establish accuracy.

### B. Accuracy and attribution against metric GT

Use the optional four GT2D/Stage3 × GTCamera/Stage1Camera lanes only when
attributing project-integration 2D-pose and camera error. For the independent
model-only claim, use `evaluate-systems` to compare v0.2 with v0.1.6
on a pairwise common pose/joint cohort.  The evaluator also emits an all-system
common-cohort table, reports each system's own coverage separately, and uses
paired pose-record bootstrap intervals for the headline deltas.

WorldPose or an equivalently calibrated soccer dataset is required for global
MPJPE, longitudinal error, GALE and ordering claims.  H3WB/SportsPose can
support articulation claims but cannot replace the pitch-world benchmark.

### C. Third-party model comparison

The recommended initializer A/B is production RTMW3D-L versus
KASportsFormer, with the same Stage-3 2D, cameras, track window and Stage-4
optimizer.  A generic MotionBERT run is an optional second 17-joint temporal
baseline.  Direct exports are converted with `convert-external` and must
declare one of:

- `PITCH_WORLD_METRIC`: global, longitudinal and offside-oriented metrics are valid;
- `ROOT_RELATIVE_METRIC`: only root-aligned and PA-MPJPE are valid;
- `SCALE_AMBIGUOUS`: only PA-MPJPE is valid.

The evaluator refuses to assign global/offside metrics to root-relative or
scale-ambiguous models.  A 17-joint third-party model is compared only on the
mapped common joints; it is never credited with nonexistent foot landmarks.

`BENCHMARK_SYSTEMS.example.json` defines the multi-system manifest.
`EXTERNAL_POSE_INPUT.example.json` and
`EXTERNAL_POSE_ADAPTER.example.json` define the external conversion boundary.

## 9. Release decision

The Stage-4 research-accuracy flag may be frozen only when all of the following
hold:

1. pipeline regression gates pass against the frozen release;
2. the independent model-only benchmark comparison is complete on real metric
   GT. The four attribution lanes are additionally required only if a project
   integration claim is made;
3. current-versus-legacy paired comparisons contain confidence intervals and
   no unexplained regression in GlobalMPJPE, longitudinal P90, GALE or close
   pair ordering;
4. uncertainty coverage is measured on held-out GT and is not called
   calibrated before a calibration procedure is evaluated;
5. third-party numbers record checkpoint hash, repository revision, dataset
   release, joint mapping and coordinate scope.
