# Stage 6 — Implementation status

## Current local patch: v0.5.1

See [V051_PATCH_NOTES.md](V051_PATCH_NOTES.md) for the separate FOOT/non-FOOT gates
and recorded validation. The v0.5.0 summary below is historical. Runtime version is
0.5.1; package metadata remains 0.4.4. No tests were rerun during the documentation cleanup.

## Historical v0.5.0 summary

Implemented locally from the conversation specification: fusion dedup/provenance,
Stage3 contact-region distances with temporal support and ambiguity rejection,
strict distal-foot Stage4 adapter, Y-plane projection/sensitivity, selected-frame-only
contact override and hybrid fallback, downstream handoff, cached refinement CLI,
synthetic regression and annotation-based metric helper.

Not validated: contact accuracy, fusion recall gain, production BLE-X accuracy.
Contact thresholds are fixed conservative heuristics, NOT calibrated on labeled contacts.
No trained contact classifier, no upper-body metric proxy and no offside verdict.
Research accuracy remains unfrozen. Older implementation notes follow.

## Implemented and exercised

- SoccerNet-v3 / SoccerNet-v3D ZIP-backed 2D benchmark path from v0.3.x.
- Optimized-vs-original GT evaluation from v0.3.6.
- Canonical SoccerNet → Stage-6 coordinate transform.
- Public YOLO candidate integration and Viterbi selection.
- Robust temporal log-diameter refinement.
- Short-gap diameter interpolation.
- Short-gap 2D centre interpolation with explicit imputation flags.
- Temporal one-range-per-ray world trajectory optimization.
- World-position second/third-difference regularization.
- Ground-compatible soft anchor.
- Pitch / height / positive-range feasibility bounds.
- Measurement-derived maximum range-deviation guardrail.
- Explicit single-frame size-prior fallback only for the user-selected `t0` when temporal evidence is insufficient; long-gap non-t0 frames remain missing.
- Cached-state temporal refinement without detector rerun.
- Runtime provenance in benchmark outputs.

## Validated here

The package contains deterministic synthetic tests demonstrating that temporal refinement reduces injected apparent-size noise/outliers, bridges short gaps, and refuses long gaps. The full historical regression suite is also run against the real Stage-1 camera workspace.

A cached 61-frame real replay was refined successfully. This confirms integration and numerical stability on the current project camera states, but does **not** establish metric accuracy because no 3D ground truth exists for that replay.

## Not claimed yet

- No real consecutive-video temporal 3D accuracy benchmark is claimed.
- No centimetre-level ball localization claim is made.
- No automatic frame-of-pass detection is introduced.
- No Stage-2 human/team/offside logic is moved into Stage 6.
- No temporal hyperparameter set is declared final; the defaults are a conservative first production configuration.

## Next empirical gate

Run `refine_ball_trajectory.py` on several representative replay windows and inspect the per-frame temporal diagnostics. The scientific validation milestone is a genuinely consecutive-frame dataset with reference ball XYZ or an equivalent strong metric reference; the SoccerNet-v3D action/replay CSV is multi-view, not a temporal sequence.
