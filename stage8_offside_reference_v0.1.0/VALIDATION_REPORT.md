# Stage 8 v0.1.0 validation report

Local package validation performed before ZIP creation:

```text
python -m pytest -q
33 passed
```

Oracle benchmark:

```text
status = PASS
oracle_logic_accuracy = 1.0
```

Example integration fixture:

```text
frame alignment = 104 / 104 / 104
preflight = READY
Stage 8 status = VALID
second-last opponent = track_011
reference source = SECOND_LAST_OPPONENT
reference X = 47.6 m
```

Optional matplotlib longitudinal QA rendering was smoke-tested successfully.

These checks establish implementation correctness on packaged tests/fixtures only. They do not establish independent metric-world or offside decision accuracy.
