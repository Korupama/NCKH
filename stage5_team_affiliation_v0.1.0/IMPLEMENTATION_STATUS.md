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

Research metrics are not frozen. Thresholds must be selected on TRAIN and reported
once on VALID/TEST.
