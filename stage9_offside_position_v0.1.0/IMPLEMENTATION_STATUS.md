# Stage 9 v0.1.0 Implementation Status

Implemented:

- deterministic ON/OFF position classification;
- LTR/RTL goalward coordinate normalization;
- opponents-half condition;
- toucher exclusion;
- Stage-4 legal-landmark and root fallback geometry;
- Stage-8 direct and reconstructed reference fallback;
- best-effort consumption of unresolved/degraded upstream artifacts;
- JSON CLI;
- local standard-library web demo;
- Stage-3 bbox/keypoint overlay support;
- Stage-4 -> Stage-1 projection fallback;
- projected Stage-8 reference line;
- oracle benchmark and unit/integration tests.

Not validated/frozen:

- end-to-end offside-position accuracy;
- legal body surface accuracy beyond Pose23 landmark proxy;
- calibrated uncertainty;
- referee/offence semantics;
- automatic frame-of-pass selection.
