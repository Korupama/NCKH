# Task Pose23@t0 data root

This directory contains only the versioned manifest/schema contract. Raw
broadcast frames and restricted annotations must remain in their authorized
external location and must not be copied into this repository.

The manifest is intentionally empty until an authorized annotation source is
selected and reviewed. Validate the template with:

```bash
cd /Volumes/Lexar/Github/NCKH/stage3_pose2d_v0.1
PYTHONPATH=. conda run --no-capture-output -n nckh-stage3-pose2d \
  python validate_pose23.py \
  --manifest data/task_pose23_t0/pose23_t0_manifest.json \
  --allow-empty
```

Every sample must provide `case_id`, `video_id`, `match_id`, `sequence_id`,
`track_id`, user-selected `frame_index`/`t0_selection`, raw-pixel bbox and
exactly 23 annotated keypoints. `split_group_id` must not cross train,
validation or test.
