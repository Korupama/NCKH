# QA — Stage 5 v0.1.1 benchmark tooling

## Automated tests

- `python -m pytest -q`
- Result: **13 passed**.

## Synthetic third-party benchmark validator

- `python validate_stage5_benchmark_v011.py`
- Result: **PASS**.
- Synthetic outfield selective accuracy: 100%.
- Synthetic outfield overall accuracy: 87.5% (one intentionally conservative UNKNOWN).
- Synthetic goalkeeper assignment: correct.
- Referee team contamination: 0%.

These synthetic numbers validate the benchmark plumbing only; they are not research performance claims.

## Real SoccerNet-GSR label-schema smoke

The supplied real `Labels-GameState.json` was parsed successfully as:

- SoccerNet-GSR version: **1.3**
- sequence: **SNGS-060**
- frames: **750** at 25 FPS
- resolution: **1920x1080**
- team-bearing tracks: **22**
- outfield player tracks: **20**
- goalkeeper tracks: **2**
- referee tracks: **3**

The corresponding image files/video were not mounted in this runtime, so no real SoccerNet team-affiliation accuracy is reported here.

## Status

- benchmark implementation: **VALIDATED**
- real third-party quantitative accuracy: **NOT YET RUN IN THIS RUNTIME**
- research accuracy frozen: **false**
