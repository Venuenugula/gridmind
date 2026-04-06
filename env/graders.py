def grade_episode(metrics):
    if metrics["outage_events"] > 1:
        return 0.0

    score = 0
    score += metrics["demand_satisfaction"] * 0.35
    score += (1 - metrics["cost_ratio"]) * 0.25
    score += metrics["renewable_usage"] * 0.2
    score += metrics["stability"] * 0.2

    return max(0, min(score, 1))