# Stage 5 implementation status

## Current implementation

- Package version: `0.2.1`.
- Legacy replay method remains the default.
- Dominant-team clustering and residual-role recovery are available as opt-in paths.
- SoccerNet-GSR v1.3 reader supports B0 (`bbox-color`), B1 (`stage5-color`) and external predictions.
- Track-level accuracy, selective accuracy, coverage, macro-F1, ARI, NMI and bootstrap confidence intervals are reported.
- Referees are excluded from team clustering.
- GK/referee residual decisions are fail-closed by default and emit reason codes.

## Current benchmark result

The local B0 `bbox-color` run on SoccerNet-GSR v1.3 `valid` completed on 58 sequences:

| Metric | Result |
|---|---:|
| Overall accuracy | 90.26% |
| Coverage | 93.29% |
| Selective accuracy | 96.75% |
| Macro-F1 | 96.60% |
| Outfield accuracy | 95.63% |
| Goalkeeper accuracy | 10.39% |
| Referee contamination | 0% |

This is an oracle-box baseline, not an end-to-end detector/tracker result. The B1
pose-guided method and the torso/lower-body fusion change still require a compatible
pose cache before they can be compared fairly.

## M2 benchmark result

The oracle-track RTMW-L pose cache and the B1 pose-guided benchmark were run on the
same 58-sequence `valid` split. M2 uses fused torso/lower-body appearance with weights
`0.75/0.25`.

| Metric | B0 bbox-color | M2 stage5-color |
|---|---:|---:|
| Overall accuracy | 90.26% | 91.08% |
| Coverage | 93.29% | 93.94% |
| Selective accuracy | 96.75% | 96.95% |
| Macro-F1 | 96.60% | 96.76% |
| Outfield accuracy | 95.63% | 96.24% |
| Outfield selective accuracy | 98.65% | 99.19% |
| Goalkeeper accuracy | 10.39% | 14.29% |
| Goalkeeper coverage | 38.96% | 48.05% |
| Referee contamination | 0% | 0% |

M2 improves the oracle-box baseline, especially for outfield players and coverage, but
goalkeeper affiliation remains the main failure mode. The result is not end-to-end:
GSR boxes, roles and track IDs are used as oracle inputs; team labels remain hidden.

Benchmark outputs:

- B0: `/home/tondaiquoc/benchmarks/stage5_valid_bbox_color/benchmark_summary.json`
- M2: `/home/tondaiquoc/benchmarks/stage5_valid_stage5_color_m2_20260924/benchmark_summary.json`

Research metrics are not frozen. Thresholds must be selected on TRAIN and reported
once on VALID/TEST.
