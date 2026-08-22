from __future__ import annotations


def risk_level(probability: float, thresholds: dict[str, float]) -> str:
    if probability < thresholds["low"]:
        return "LOW"
    if probability < thresholds["medium"]:
        return "MEDIUM"
    if probability < thresholds["high"]:
        return "HIGH"
    return "CRITICAL"


def priority_score(
    probability: float, monthly_charges: float | None, assumptions: dict[str, float]
) -> float:
    value = float(monthly_charges or 0.0) * 12.0
    value_factor = min(max(value / max(assumptions["customer_value"], 1.0), 0.25), 2.0)
    return round(float(min(100.0, probability * 100.0 * value_factor)), 2)
