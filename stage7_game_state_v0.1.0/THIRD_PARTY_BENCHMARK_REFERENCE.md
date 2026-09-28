# Third-party benchmark comparison used for Stage 7 targets

## SoccerNet Game State Reconstruction (CVPRW 2024)

SoccerNet-GSR contains role, team affiliation/team side, track identity and pitch localization. The original full baseline reports GS-HOTA 22.26. Its module ablation reports 92.00 GS-HOTA for Team Side when the other relevant modules are replaced by ground-truth oracles.

Use: role/team/referee context reference.
Do not use: direct Stage-7 attacking-team, toucher or attack-direction GT.

## SoccerNet GSR Challenge 2025

The challenge results report the top GS-HOTA increasing to 63.90. This is a full-pipeline score and is not numerically equivalent to any Stage-7 exact-match metric.

Use: evidence that modern full GSR remains materially below perfect end-to-end reconstruction.

## Broadcast2Pitch (WACV 2026)

On SoccerNet-GSR test, Broadcast2Pitch reports GS-HOTA 61.48, GS-AssA 78.00 and IDF1 64.20. Its attribute ablation reports 79.52 for pitch-only, 79.20 for pitch+role, 73.58 for pitch+team, and 64.70 for pitch+role+team.

Use: evidence that role can be relatively tractable while team attribution can still create a significant full-pipeline penalty.

## FOOTPASS (CVIU / SoccerNet 2026)

FOOTPASS predicts player-centric events represented by frame, team, jersey/player and action. The strongest official validation baseline TAAD+DST is reported at Micro F1 0.675; a later ME-DST preprint reports 0.778. This task is harder than Stage 7 because it jointly searches who/what/when over full match video, whereas the project manually fixes t0.

Use: a related lower-bound reference for actor/toucher and team attribution targets.
Do not use: a direct threshold for Stage-7 direction or referee exclusion.

## Resulting internal targets

The v2 `pilot-minimum` profile intentionally uses a toucher accuracy floor of 0.70, attacking-team accuracy 0.85, participant-set micro-F1 0.90, referee-exclusion F1 0.90, full exact match 0.50, and invariant pass rate 1.00. Direction accuracy 0.90 is an internal-only target because no directly equivalent public metric was found.

The `research-target` profile raises those values after the pilot is frozen. These targets are not presented as official public standards.
