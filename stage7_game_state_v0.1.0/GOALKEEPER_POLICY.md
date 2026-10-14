# Fixed goalkeeper membership

`GOALKEEPER_ALWAYS_DEFENDING` is the requested project policy: every active
track recognized as goalkeeper / keeper / GK belongs to the defending set,
regardless of its previous appearance cluster, position or jersey colour.

Stage 5 (legacy and residual pipelines) no longer guesses a goalkeeper's cluster
from appearance, spatial medians, defensive tails or defended-half statistics.
It emits `team_side=DEFENDING`, `team_id=null`, `team_status=PENDING_CONTEXT`.
Stage 7 assigns the opposing cluster ID when the attacking cluster and exactly
one opposing outfield cluster are known. Membership in `opponents` does not
depend on having that numeric ID. Resolved assignments and original IDs are
recorded in `diagnostics.goalkeeper_assignments` and consumed by the web demo.

Stage 8 ranks these goalkeepers with all other defenders using the same legal
body landmarks: a goalkeeper can be first, second, or further down the ranking.
No rank is forced. Missing geometry retains the existing partial-data rules.
Stage 9 never classifies a goalkeeper as an attacker. Stale artifacts that place
a known goalkeeper in attackers require Stage 7 and Stage 8 to be rerun; changing
only the overlay would leave the wrong reference line.

The fixed policy also applies when multiple goalkeepers are detected. If a
goalkeeper is the toucher, Stage 7 reports an unresolved policy conflict rather
than promoting that goalkeeper to attacker. This is a project assumption, not a
general football rule. Historical benchmark ground truth is not rewritten:
legacy oracle cases with an attacking-team goalkeeper deliberately disagree
with this policy. Old benchmark scores are not results of this new version.

Bookmark runs and web inference use this policy automatically. Existing saved
JSON and images are immutable results of their original run and are not patched.
