# Pose23@t0 annotation guideline v1.0

## Purpose

Create a task-aligned evaluation set for Stage 3 in broadcast football
scenes. Each sample represents one player track at the user-selected
frame-of-pass `t0`. This dataset evaluates 2D evidence quality only; it does
not label team, attack direction, ball state, legal-body membership or an
offside decision.

## Unit of annotation

One sample is:

```text
case_id + video_id + match_id + sequence_id + frame_index=t0 + track_id
```

The source image must be the original broadcast frame in
`RAW_DISTORTED_PIXEL` coordinates. Do not resize coordinates into a model
input space. Store the original `image_width`, `image_height` and player
`bbox_xyxy` as `[x1,y1,x2,y2]`.

The user selects `t0`; annotators do not infer a new frame-of-pass. If the
same player is annotated in nearby frames for temporal analysis, all samples
from that match/sequence remain in the same split.

## Keypoints

Annotate exactly these 23 points in canonical order:

```text
17 COCO body: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
6 feet: left/right big toe, small toe and heel
```

Coordinates are pixel positions in the original frame. Do not mirror left and
right. “Left” and “right” refer to the player's anatomical left/right, not
the viewer's screen side.

For a visible or partially occluded landmark, place the best estimated
anatomical location in the image and use the corresponding visibility label.
For a landmark outside the image or intentionally not annotated, set `x/y`
to `null`.

## Visibility and review labels

Each keypoint uses exactly one label:

- `VISIBLE`: landmark is visible enough for direct localization.
- `OCCLUDED`: landmark is hidden by another player/object, but its location
  can be estimated from visible context; keep the estimated pixel coordinate.
- `TRUNCATED`: the player/body part is cut by the image border; keep a point
  only if its location remains inside the image and annotate the border slice.
- `OUT_OF_FRAME`: no valid pixel location exists; `x/y` must be `null`.
- `NOT_ANNOTATED`: source quality or identity is insufficient; `x/y` must be
  `null` and the sample is excluded from the primary metric.

`annotation_confidence` is a 0–1 annotation certainty score, not model
confidence. Use `review_status` per point and `annotation_review_status` per
sample: `SINGLE_ANNOTATOR`, `DOUBLE_REVIEWED` or `ADJUDICATED`.

## Scene tags

Required tags:

- `player_pixel_height`: bbox height in original pixels;
- `scale_bin`: `SMALL`, `MEDIUM`, `LARGE`, or `UNKNOWN`;
- `occlusion_level`: scene/player occlusion severity;
- `motion_blur`: `NONE`, `MILD`, `SEVERE`, or `UNKNOWN`;
- `view`: `FRONT`, `BACK`, `SIDE`, `THREE_QUARTER`, or `UNKNOWN`;
- `border_truncated`: whether the player bbox touches/crosses the image border;
- `crowd_level`: `ISOLATED`, `MODERATE`, `CROWDED`, or `UNKNOWN`.

Recommended operational bins should be frozen before annotation starts. Do not
change bins after seeing model results.

## Split policy

Use `split_group_id`, recommended as `match_id`. A match or match/sequence
group must occur in exactly one of train, validation or test. Never split
adjacent frames from one sequence across partitions. Keep the test split
locked before threshold or model tuning.

The validator checks both `split_group_id` and `(match_id, sequence_id)` for
cross-split leakage and duplicate `(match, sequence, frame, track)` records.

## Evaluation policy

Primary metric: `VISIBLE` points.

Secondary diagnostic: `OCCLUDED` points.

Report `TRUNCATED`, `OUT_OF_FRAME` and `NOT_ANNOTATED` as coverage/failure
slices, not as ordinary visible-point accuracy. Stage 3 temporal estimates
must never be scored as observed raw `t0` predictions.

Before annotation use, obtain and record the source license/permission and
keep restricted media outside the repository.
