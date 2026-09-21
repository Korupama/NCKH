def ground_quality_reasons(metrics: dict, coverage: float | None) -> list[str]:
    reasons = []
    if coverage is None or coverage < 0.50:
        reasons.append("ground_anchor_coverage_at_t0 < 0.50")
    residual = metrics.get("ground_contact_residual_cm", {})
    for key, limit in (("median", 25), ("p95", 75)):
        value = residual.get(key)
        if value is None:
            reasons.append(f"ground_contact_residual_{key}_cm unavailable")
        elif value > limit:
            reasons.append(f"ground_contact_residual_{key}_cm > {limit}")
    return reasons
