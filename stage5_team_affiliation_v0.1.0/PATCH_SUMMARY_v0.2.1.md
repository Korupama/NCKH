# Stage5 v0.2.1 patch summary

This is a corrective, fail-closed patch derived from the five-sequence v0.2.0
smoke result. It does not claim improved real-data accuracy until a new smoke run
is completed.

Corrections:

- pairwise GK ordering only compares residuals that independently pass the
  observation-count and goal-distance candidate gates;
- unrelated same-half appearance outliers are recorded as ignored diagnostics
  and cannot veto a goalkeeper;
- pairwise ordering among qualified goal candidates is retained as a diagnostic,
  not a hard rejection by default; a frozen TRAIN config may explicitly require it;
- non-GK residuals remain `unknown_residual` by default; the weak v0.2.0 referee
  heuristic is available only through explicit frozen-config opt-in;
- defensive-tail GK team association is calculated for diagnostics but does not
  produce a VALID team by default;
- outputs include explicit decision reason codes and pairwise gate diagnostics.

Legacy V0 remains the default replay method. No replay or long GSR benchmark was
run while applying this patch. Research accuracy remains unfrozen.
