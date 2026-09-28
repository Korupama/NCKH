# Stage 6 Benchmark v1.4.2 preflight hotfix

Overlay `stage6_ball_localization_v0.4.4/` on the existing Stage 6 package.

This hotfix adds `contact-preflight` and prevents a frozen FOOTPASS manifest from being built from the literal template placeholder `relative/path/to/...`.

Run:

```powershell
python benchmark_stage6_protocol.py contact-preflight --manifest ".\\data\\stage6_footpass_manifest.json"
```

Only run `contact-production` after prediction artifacts are bound and preflight reports `READY`.
