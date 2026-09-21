# Stage 6 v0.5.1 — contact geometry gate hotfix

Local implementation of the requested conversation specification; folder name stays v0.4.4.
Supersedes v0.5.0's shared ground-anchor/camera gate and its requirement to supply
--stage4-handoff for every contact-aware invocation.

- FOOT: no Stage4 anchor required/read/used. Forward ray intersection at Z=ball radius,
  finite coordinates, plausible height and pitch bounds are still required.
- Camera VALID or DEGRADED is allowed for FOOT, with corresponding VALID_CONTACT_GROUND_PLANE
  or DEGRADED_CONTACT_GROUND_PLANE status. Top-level quality is not upgraded.
- Non-FOOT: requires VALID camera and VALID foot-derived anchor. No degraded-camera override.
- Association supports the existing SUPPORTED semantic; association/tracking thresholds unchanged.
- Separate fallback reasons for ground camera, vertical camera, anchor, and infeasible geometry.
- localization exposes ground_anchor_required, ground_anchor_used and camera_status.
- v051_gates implementation=VALIDATED refers to release tests, not tests run during inference.
  Contact support is case-level; metric accuracy remains unvalidated and research unfrozen.

Regression with STAGE1_ROOT configured: 65 passed. Hotfix-specific validator: 10 cases.
No new SDK/model, SAM3D dependency, detector inference or Viterbi rerun.

Cached CLI example (FOOT requires no Stage4 argument):

```powershell
python refine_ball_contact.py --state-json outputs/replay_t0_104_cached_v050/stage6_hybrid/ball_trajectory_state.json --stage1-root D:/NCKH/stage_1_camera_v12 --stage3-state outputs/replay_t0_104_cached_v050/stage3/tracked_pose_2d_state.json --output-dir outputs/replay_t0_104_contact_v051
```

Observed real replay: track_006 / FOOT / SUPPORTED, DEGRADED_CONTACT_GROUND_PLANE,
fallback=false, anchor required/used=false, XYZ [40.4590668732,19.1464091353,0.11].
Current camera ground-only estimate independently matches these coordinates. Historical
cache ground XYZ is [40.4683646864,19.1904958035,0.11], delta X=-0.0092978133 m.
The cached point projects to [722.349334,425.367262] under the current camera, whereas
the current observation is [722.935791,425.592834]. Thus zero delta is not assumed.
Camera path/SHA256 is recorded for reproducibility. This is not an accuracy improvement claim.
