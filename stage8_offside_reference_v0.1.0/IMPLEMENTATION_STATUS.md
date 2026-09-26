# Stage 8 v0.1.0 implementation status

Implemented:

- Stage-4/6/7 adapters for current downstream handoff schemas
- legal Pose23 landmark proxy
- attack-direction-normalized `q=sX` geometry
- opponent ranking and tied second-last identity handling
- ball goalward extent
- final ball-vs-second-last reference plane
- strict fail-closed preflight
- deterministic oracle benchmark
- partial-GT evaluation manifest support
- optional longitudinal QA plot

Not claimed:

- independent metric-world accuracy
- exact body-surface accuracy
- calibrated Stage-4 uncertainty propagation
- public second-last-opponent benchmark completion
- Stage-9 offside-position accuracy

Production integration at project frame 104 remains gated by a `VALID` Stage-7 context.
