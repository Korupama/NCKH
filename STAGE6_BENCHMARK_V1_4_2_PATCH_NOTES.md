# v1.4.2

- Adds `contact-preflight`.
- Detects template placeholder paths, missing/invalid Stage-6 state JSON, missing/invalid track identity maps, and missing GT needed for each metric family.
- Frozen FOOTPASS manifest creation now rejects the literal template placeholder state path.
- Does not change Stage-6 runtime inference or previously defined metrics/thresholds.
