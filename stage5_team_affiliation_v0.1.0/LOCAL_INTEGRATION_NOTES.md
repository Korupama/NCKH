# Local Stage 5 integration — 2026-09-19

Work was applied directly to the extracted Stage5 v0.1.0 artifact. The separate
implementation previously prepared on C: is not used.

Changes:
- QUICKSTART now points to the existing Stage2/Stage3 index-104 replay inputs.
- Stage2/Stage3 consistency check rejects missing Stage2 track IDs.
- The configured minimum lower-body sample count is applied to both goalkeeper
  evidence and the outfield lower-body reference features.
- Existing output directories are rejected before processing to preserve results.
- Failure to decode the selected frame stops processing without writing outputs.
- Four regression tests were added for these guards.

Validation: source review and input JSON inspection only. Tests and real replay
inference were NOT RUN at the user's request. Existing packaged test reports are
not evidence that these local edits pass. Run the commands in QUICKSTART yourself.

The index-104 Stage3 input contains 10 tracks (9 labeled player, 1 goalkeeper).
track_003 is still labeled player upstream despite the previous referee review.
No role labels, input files, old outputs or Stage4/Stage6 source were modified.
Stage5 does not determine toucher, attacking team, attack direction or offside.
