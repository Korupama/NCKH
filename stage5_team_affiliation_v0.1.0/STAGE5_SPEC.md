# Stage 5 Contract — v0.1.0

## Mission

Produce track-level team affiliation for players visible in the replay analysis window.

## Team semantics

`team_id` is one of:
- `0`
- `1`
- `null`

`0/1` are permutation-invariant internal cluster IDs. They are **not** home/away, left/right, attacking/defending.

`team_status`:
- `VALID`: usable downstream;
- `UNKNOWN`: insufficient/conflicting appearance evidence;
- `NOT_APPLICABLE`: referee or unsupported human role.

## Outfield player policy

- Extract torso appearance from shoulders/hips when pose is usable.
- Fallback to an internal bbox torso region when configured.
- Remove obvious green-pitch pixels and extreme dark/bright pixels.
- Aggregate frame features per track by robust median.
- Fit KMeans with `k=2` only on role=`player` tracks with enough usable torso frames.
- Fail closed to `UNKNOWN` for low-margin/outlier tracks.

## Goalkeeper policy

Goalkeepers are not used to fit outfield clusters. v0.1 assigns them only when lower-body appearance has a sufficiently clear affinity to one of the two outfield-team lower-body centroids. Otherwise `UNKNOWN`.

This is intentionally conservative; v0.1 does not infer goalkeeper team from tactical assumptions.

## Referee policy

Role=`referee` is excluded before clustering and receives `NOT_APPLICABLE`.

## Stage boundaries

Stage 5 does not consume Stage 4 and does not resolve attacking team. Stage 7 combines Stage 5 team IDs with Stage 6 toucher identity.
