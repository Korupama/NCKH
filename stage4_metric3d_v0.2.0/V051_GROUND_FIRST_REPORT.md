# Stage 4 v0.5.1: ground-first refinement

Implemented locally from the latest proposal in the referenced conversation `rút gọn`.
The referenced hotfix ZIP was not available locally; this is not a byte-identical ZIP application.

## Changes

- Ground validity no longer depends on proximity to SAM translation.
- RTMW foot rays intersect Z=0; candidates require supported pitch origin, pitch bounds (2 m margin), and no other primary foot below -0.10 m.
- Confidence-weighted consensus uses a 0.75 m neighborhood and coordinatewise median; conflicting groups without majority support are rejected. A single geometrically valid candidate remains usable.
- Valid ground consensus initializes refinement, with 0.10 m ground sigma and weak 10 m SAM-prior sigma. Without usable ground, SAM initialization/prior remains the fallback.
- Grounded refinement is bounded to 0.75 m Euclidean distance using conservative per-axis bounds of 0.75/sqrt(3).
- Diagnostics include candidate spread, consensus support, SAM disagreement, ground/refined displacement and temporal status. Top-down plots distinguish RTMW ground hits, SAM translation origins and refined translation origins (these origins are not anatomical hip centers).
- Geometry gate fails for ground coverage below 50%, median contact residual above 25 cm, or P95 above 75 cm; missing residual measurements also fail. These are sanity thresholds, not validated accuracy requirements.
- Historical backends and old outputs are retained. No SAM inference was repeated for this hotfix.

## Real cached experiment: image 105 / zero-based frame 104

Same native SAM3D cache, Stage-3 observations and Stage-1 cameras. One-frame run, temporal regularization disabled. Coverage below uses the 10 observations with valid SAM priors; 3 of 13 tracks still lack priors (001, 002, 010).

| Metric | v0.5 refined | v0.5.1 refined |
|---|---:|---:|
| Ground-anchor coverage among valid priors | 20% | 100% |
| Ground residual median (cm) | 46.35 | 5.04 |
| Ground residual P95 (cm) | 135.05 | 14.89 |
| Reprojection median (px) | 2.08 | 3.73 |
| Reprojection P95 (px) | 20.90 | 32.42 |
| Below-pitch fraction | 3.48% | 1.74% |

New refined gate: PASS_SANITY. Direct baseline fails the ground-residual gate.
Ground-vs-SAM translation disagreement median/P95: 4.72/11.89 m.
Ground-to-refined displacement median/P95: 0.067/0.414 m.
Keeper track016 hip-root world coordinates: approximately [50.71, 3.31, 0.70] m, within the right goal-area footprint for this camera/pitch convention.

Ground consistency improves substantially, but image reprojection worsens. Neither quantity is independent metric GT: the ground constraint and reprojection observations are also optimization inputs. This result does not establish true 3D accuracy or offside decision accuracy. Single-frame data cannot validate temporal smoothing. Jumping/non-contact poses and inaccurate camera/foot observations remain important failure cases.

## Reproduce

Run `powershell -ExecutionPolicy Bypass -File .\rerun_frame104_v051.ps1` from this workspace. It reuses the native cache and writes only `runs/stage4_v051_frame104`, not the previous v0.5 experiment folder. Re-running replaces files in the v0.5.1 output folder.

Inspect `runs/stage4_v051_frame104/sam3d-pitch-refined/stage4_quality_report.json` and its top-down image. Execution success alone is not a quality-gate pass.

## Remaining evaluation

Historical local automated verification: `48 passed` across `tests`, `tests_v04`, and `tests_v05` (49.94 s). Current Phase-0 verification is `50 passed` in `nckh-env`, including provenance and headless visualization tests. The suite includes a synthetic SAM translation offset of (4, -3, 5) m, regression rejection of old frame-104 contact metrics, and missing-metric failure behavior.

Metric GT and Stage-8/9 evaluation remain NOT_EVALUATED; `research_accuracy_frozen = false`. Next evaluate independent clips with known pitch positions/contact labels, include airborne/contact ambiguity, and run a multi-frame cache for temporal validation. Do not tune on a held-out test set.
