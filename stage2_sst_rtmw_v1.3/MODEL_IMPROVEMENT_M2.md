# Stage 2/5 improvement M2

## Changes

### Stage 2: motion-aware target-window association

The Stage-2 tracker now adds a short-horizon constant-velocity prior to the Hungarian
cost after an anchor has two accepted observations. The prior is bounded by box height
and three frames; it cannot create a track or change the selected-frame anchor set.

Configuration:

```text
use_motion_prior=true
motion_weight=0.15
```

The intended benchmark comparison is:

```text
baseline: --no-motion-prior
M2:      --motion-weight 0.15
ablation: 0.05, 0.25
```

Report `IDF1`, `HOTA`, `AssA`, `DetA`, `TCR`, `AnchorCoverage`,
`ConditionalTCR`, `CandidateRecall`, and `RefereeLeakageRate` for every setting.

External MOT baselines such as ByteTrack and BoT-SORT should only be reported after
their weights and input protocol are reproduced on the same split.

### Stage 5: fused multi-region appearance

Stage 5 now fuses torso and lower-body appearance for outfield players. Missing regions
are renormalized:

```text
fused = normalize(0.75 * torso + 0.25 * lower)
```

If a region is unavailable, its weight is removed and the remaining feature is
renormalized. Referees are still excluded before clustering; goalkeeper assignment
remains conservative and lower-body based.

Benchmark the following on `valid`, then freeze thresholds on `train` only:

```text
B0: bbox-color
B1: torso-only Stage 5
M2: torso+lower fusion (0.75/0.25)
M2-ablate: 0.90/0.10, 0.75/0.25, 0.60/0.40
```

Report track accuracy after per-sequence team-ID permutation, macro/per-team F1,
ARI, NMI, unknown rate, goalkeeper accuracy, referee leakage, and selected-frame
team coverage. Do not claim an improvement until the same sequences and hidden-team
protocol are used for all rows.

## Reproducible commands

Install the package requirements and test dependencies in each stage folder, then:

```bash
python -m pytest -q stage2_sst_rtmw_v1.3/tests
python -m pytest -q stage5_team_affiliation_v0.1.0/tests
python stage2_sst_rtmw_v1.3/benchmark_stage2.py run \
  --soccernet-root "$SOCCERNET_GSR" --split valid --protocol quick \
  --sst-checkpoint "$SST_CHECKPOINT" --rtmw-model "$RTMW_MODEL" \
  --output-dir stage2_benchmark_m2
python stage5_team_affiliation_v0.1.0/benchmark_soccernet_gsr.py \
  --soccernet-root "$SOCCERNET_GSR" --split valid --method stage5-color \
  --summary stage5_benchmark_m2.json
```

The Stage-5 oracle-track RTMW-L cache and B1 benchmark were completed on the 58-sequence
SoccerNet-GSR v1.3 `valid` split. M2 improved overall accuracy from 90.26% to 91.08%,
outfield accuracy from 95.63% to 96.24%, and goalkeeper accuracy from 10.39% to
14.29%. The goalkeeper result remains insufficient for a final claim.

The repository does not contain SST/RTMW weights. Stage 2 model metrics still require
those weights; Stage 5 B0 runs without a pose cache, while B1 requires one.
