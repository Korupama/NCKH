# Stage 6 v0.5.0 local patch

Implemented from the latest conversation specification; not an extraction of the
ChatGPT-generated archive. Keep the installation folder name v0.4.4.

## Safety and selection policy

- Only VALID Stage3 keypoints enter contact geometry. Other quality states are excluded.
- Nearest region must be within ball radius in pixels + 4 px; another player within
  3 px of the best eligible distance makes the result ambiguous.
- Temporal evidence needs valid observations on both sides within 3 frames, distance
  rise >1 px on both sides, and t0 within 2 px of the local minimum. This is only a heuristic.
- Contact requires VALID distal-foot ground anchor, valid camera, finite forward ray,
  pitch bounds, broad region-specific height bounds, and <=1.5 m horizontal anchor offset.
- FOOT intersects Z=ball radius; other regions intersect Y=ground anchor Y. No fixed
  head/shoulder/hip height is imposed. ARM_HAND remains a region label, not legal-body evidence.
- Without support: hybrid cache, valid size prior if needed, otherwise missing.
- Only selected frame geometry changes; t0 neighborhood is evidence, not a rewritten trajectory.

## Frame 86 initial smoke

Inputs: stage6_v044_cached_replay, Stage3 stage3_real_1920, Stage4 v031_quality_gated.
Nearest usable pose region is track_007 FOOT, distance 79.893 px; threshold 9.790 px.
Result: NO_CONTACT_EVIDENCE, no assigned toucher, explicit hybrid fallback.
X=37.702010 m, Y=19.455678 m, Z=0.110000 m; delta X vs cache=0.
This does not establish whether the selected ball candidate or t0 is correct.
No contact sensitivity is reported because no contact plane was accepted.

## Dependencies and verification

Final installed regression with STAGE1_ROOT=D:\NCKH\stage_1_camera_v12:
55 passed, 0 failed, 0 skipped. Without that variable: 52 passed, 2 skipped.
Original modified files are backed up under backups/before_v050_20260918_060823.

Existing NumPy/OpenCV, pytest for tests, Ultralytics only for new YOLO inference.
No pretrained contact model and no extra model download. Cached refinement does not
run YOLO, fusion or Viterbi. These paths have separate synthetic tests.
Implementation success must not be represented as accuracy or research freeze.
