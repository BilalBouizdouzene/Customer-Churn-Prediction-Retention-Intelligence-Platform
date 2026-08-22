import numpy as np

from churn_intelligence.decisions.segmentation import priority_score, risk_level
from churn_intelligence.evaluation.cost_analysis import (
    business_cost_table,
    select_operating_threshold,
)
from churn_intelligence.evaluation.threshold import threshold_table

ASSUMPTIONS = {
    "customer_value": 1200.0,
    "retention_offer_cost": 80.0,
    "contact_cost": 5.0,
    "churn_loss": 1200.0,
    "retention_success_rate": 0.25,
}


def test_risk_segments_and_priority():
    assert risk_level(0.1, {"low": 0.25, "medium": 0.5, "high": 0.75}) == "LOW"
    assert risk_level(0.9, {"low": 0.25, "medium": 0.5, "high": 0.75}) == "CRITICAL"
    assert 0 < priority_score(0.7, 90, ASSUMPTIONS) <= 100


def test_threshold_and_cost_tables():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    thresholds = threshold_table(y, p)
    costs = business_cost_table(y, p, thresholds["threshold"], ASSUMPTIONS)
    assert len(thresholds) > 10
    assert {"expected_net_value", "customers_contacted"}.issubset(costs.columns)
    threshold, reason = select_operating_threshold(costs, capacity=3)
    assert 0.1 <= threshold <= 0.9
    assert "capacity" in reason
