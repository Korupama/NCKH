# Stage5 v0.2.0 — Dominant teams and residual role recovery

In-place opt-in implementation in the existing v0.1.0 directory. No pretrained
model, no new dependencies, no Stage4 metric proxy. Old replay outputs and V0
benchmark results are preserved. Default run_stage5.py remains legacy-v0 until
the oracle experiment has met its research gates.

## Methods

| Method | Role input | Geometry | GK affiliation |
|---|---|---|---|
| bbox-color (V0, unchanged) | GT role | none | lower-body colour |
| residual-v1 | hidden | none | UNKNOWN |
| residual-v2 | hidden | oracle bbox_pitch, median depth | UNKNOWN |
| residual-v3 | hidden | oracle bbox_pitch, temporal rank/order | defensive top-2 tail |

All residual arms use a trimmed K=2 with 80% cores (minimum two tracks/team).
They reject far-from-both appearances; low-margin ambiguity and unavailable
appearance are UNKNOWN, not automatically residual or referee. Core radius is
capped by prototype separation. Degenerate clusters fail closed.

V3 considers at most one GK per goal side. Defaults: >=5 observations, distance
to goal line <=18 m, Top2Rate>=0.60 on frames with >=3 humans, pairwise ordering
>=0.70 with >=5 overlapping frames for each same-side residual competitor.
Ties/insufficient support remain UNKNOWN. If there is no same-side competitor,
the all-human rank and distance gates still apply. Split/fragmented GK tracks and
two simultaneous GKs at one goal are limitations, not solved by this first rule.

Referee requires a far-from-goal residual (>22 m), low Top2Rate on BOTH sides
(<=0.20), and persistent depth ordering behind an accepted GK. A lone ambiguous
residual is not forced to referee. Roaming is only approximated by non-goal-anchored
temporal ranks, not a learned movement model.

GK team uses simultaneous outfield top-2 median depth for each team. It requires
>=5 frames, median tail separation >=2 m and >=70% consistent, separated votes.
No lower-body-colour fallback in V3. Missing geometry never invents positions.

## Appearance filter difference

The HSV/Lab descriptor remains unchanged, but new residual arms preserve dark,
green and achromatic kit pixels (S>=0, 8<=V<=255; no green removal). V0 is untouched.
This prevents black referee/green GK/white player observations being discarded by
the v0.1 pitch-colour filter. Bbox contamination remains a limitation. Consequently
V0 versus V1 changes both robust clustering AND pixel selection; it must not be
reported as a pure clustering-only ablation. V1/V2/V3 share identical extraction.

## Leakage boundary and geometry convention

Benchmark input is GT human bbox/track ID. GSR category IDs are collapsed to HUMAN
for population selection; ball/pitch/camera/other are excluded. Fine role/team,
jersey and other semantic attributes are stripped before the predictor. This is
oracle-human detection, not end-to-end Stage2 evaluation. Labelled categories are
not passed to clustering/role recovery. GT semantics are read by evaluation only.

Oracle coordinates use bbox_pitch.x_bottom_middle/y_bottom_middle, pitch origin at
centre, X goal-to-goal in metres, goal lines +/-52.5. Missing/nonfinite/out-of-pitch
geometry is unusable and reported via UNKNOWN support. This is NOT a test of
Stage1 camera accuracy. Neither pose inference nor camera estimation runs here.

## Commands for the user (NOT run by the assistant)

```powershell
cd D:\NCKH\stage5_team_affiliation_v0.1.0
$env:OMP_NUM_THREADS = "1"

# Optional metadata refresh after the in-place version change
python -m pip install -e .
python -m pytest -q

# First: oracle smoke on 5 sequences, new output directory
python -u benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" --split valid --method residual-v3 --limit 5 `
  --output-dir ".\benchmark_results\gsr_v020_v3_5seq" --progress-every 1

# Full available VALID split; do not retune defaults using these results
python -u benchmark_soccernet_gsr.py run `
  --gsr-root "D:\GSR" --split valid --method residual-v3 `
  --baseline-summary ".\benchmark_results\gsr_valid_bbox_color_full_manual\benchmark_summary.json" `
  --output-dir ".\benchmark_results\gsr_v020_v3_valid" --progress-every 1
```

Verify the V0 baseline path if your full-run directory differs. For smoke comparison
use a baseline containing the exact same five sequences, not the full 58-sequence
summary. The adjacent sequence_metrics.jsonl is required for paired gate checks.

Repeat with --method residual-v1 and residual-v2 in different output directories.
Use the existing compare command for team metrics; role P/R/F1 and contamination
are additionally in each benchmark_summary.json/.md. Every new run refuses an
existing output directory. run_manifest.json distinguishes RUNNING/FAILED/COMPLETE;
an incomplete run never writes a completed aggregate summary. This release does
not resume checkpoints automatically.

Tune only TRAIN via --residual-config frozen_config.json (ResidualConfig fields);
freeze before VALID. No automatic search or VALID-based tuning has been performed.
The manifest stores all thresholds; sequence metrics include label SHA256.

## Metrics and gates

Team labels align per-sequence using outfield GT only, then that mapping is reused
for GK. UNKNOWN counts as wrong in overall accuracy, excluded in selective accuracy.
No outfield alignment evidence => affiliation scoring becomes UNKNOWN. Role P/R/F1
includes all GT humans, UNKNOWN as FN. Report player->GK and player->referee counts
and rates separately. GK-only ARI/NMI are suppressed. Existing team F1/ARI/NMI remain
assigned-only for comparability and must not be confused with overall metrics.

Research targets: outfield accuracy drop <0.5 percentage point vs paired V0,
GK team overall >=80%, GK role F1>=90%. Targets are not promises. Without a paired
baseline the combined gate is NOT_EVALUATED. research_accuracy_frozen remains false
even on a gate pass: the next step is predicted geometry evaluation, not auto-freeze.

## Replay adapter, after oracle gates

run_stage5.py supports --method residual-v1/v2/v3. V2/V3 require --pitch-state.
Live Stage1 batch execution/cache construction is intentionally deferred until
oracle results pass, as specified in the referenced proposal. The callable seam
project_track_to_pitch(observations, intersect_pitch) accepts the Stage1 authority's
intersection callback and prefers distal feet over bbox-bottom centre.

Pitch-cache contract (not an automatically generated file):

```json
{
  "schema_version": "stage5-pitch-tracks-1.0",
  "coordinate_space": "PITCH_METERS_X_LONGITUDINAL_CENTER_ORIGIN",
  "source": "STAGE1_GROUND_PROJECTION",
  "replay_context": {
    "video_id": "download", "selected_frame": 104,
    "image_width": 1920, "image_height": 1080, "fps": 30
  },
  "tracks": [{"track_id": "track_006", "observations": [
    {"frame_index": 104, "xy_pitch_m": [40.0, 20.0], "status": "VALID"}
  ]}]
}
```

Values above illustrate the schema, not measured geometry; never run on fabricated
cache data. Real replay needs multiple co-observed tracks/frames. VALID projected
observations only; camera quality decisions belong to the Stage1 adapter.

v2 output schemas: team-affiliation-state-2.0 / stage5-downstream-handoff-2.0.
Preserve upstream_role/confidence; add stage5_role/status/method, appearance_group,
goal_context, assignment_method and effective_role (refined only if VALID).
UNKNOWN team stays null even if effective_role falls back to upstream. Stage7 must
check both statuses and research gates; Stage5 still does not infer attacking team,
attack direction, offside position, or decide the toucher.
