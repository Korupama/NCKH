# Stage 5 v0.1.0 QA

## Implementation validation

- `pytest`: **10 passed**
- `validate_stage5_v010.py`: **PASS**
- editable install: verified with `pip install -e . --no-deps --no-build-isolation`
- runtime dependencies used: NumPy, OpenCV, scikit-learn
- direct pretrained model: **none**

## Synthetic end-to-end smoke

Synthetic replay contains:
- 4 red outfield players;
- 4 blue outfield players;
- 1 goalkeeper with different torso kit but red-compatible lower body;
- 1 referee.

Observed result:
- red players: one common team cluster;
- blue players: the other team cluster;
- goalkeeper: assigned by lower-body affinity;
- referee: excluded / `NOT_APPLICABLE`;
- `SelectedFrameTeamCoverage = 1.0`.

This is an implementation smoke, not a research accuracy claim.

## SoccerNet-GSR schema smoke

A supplied real `Labels-GameState.json` was parsed successfully:
- 750 images;
- 14,290 annotations;
- 22 track-level team labels;
- 11 `left`, 11 `right`.

This validates team-label schema compatibility only. No Stage 5 inference was run on the corresponding real images in the packaging environment.

## Accuracy status

- real SoccerNet-GSR team accuracy: **NOT_EVALUATED**
- goalkeeper team accuracy: **NOT_EVALUATED**
- project replay accuracy: **NOT_EVALUATED**
- research accuracy frozen: **FALSE**
