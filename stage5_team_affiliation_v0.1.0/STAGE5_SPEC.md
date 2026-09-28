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

Goalkeepers are not used to fit outfield clusters. With a Stage 4 handoff, v0.1
first uses goal-to-goal pitch position, then image-space distance to the two
outfield clusters. Lower-body appearance remains the fallback. Every branch has
an ambiguity threshold; otherwise the result is `UNKNOWN`.

This is intentionally conservative and does not equate the goalkeeper role with
a specific team without spatial or appearance evidence.

## Referee policy

Role=`referee` is excluded before clustering and receives `NOT_APPLICABLE`.

## Stage boundaries

Stage 5 may consume the Stage 4 selected-frame handoff to resolve goalkeeper
affiliation with pitch/image spatial evidence. It still does not resolve the
attacking team. Stage 7 combines Stage 5 team IDs with Stage 6 toucher identity.
