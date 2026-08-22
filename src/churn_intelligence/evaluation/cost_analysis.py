from __future__ import annotations

import numpy as np
import pandas as pd


def business_cost_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: list[float] | np.ndarray,
    assumptions: dict[str, float],
) -> pd.DataFrame:
    offer = float(assumptions["retention_offer_cost"])
    contact = float(assumptions["contact_cost"])
    loss = float(assumptions["churn_loss"])
    success = float(assumptions["retention_success_rate"])
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        contacted = int(predicted.sum())
        true_reached = int(((y_true == 1) & predicted).sum())
        false_positive = int(((y_true == 0) & predicted).sum())
        expected_retained = true_reached * success
        cost = contacted * (offer + contact)
        avoided_loss = expected_retained * loss
        rows.append(
            {
                "threshold": float(threshold),
                "customers_contacted": contacted,
                "true_churners_reached": true_reached,
                "false_positive_contacts": false_positive,
                "expected_retained_customers": expected_retained,
                "expected_retention_cost": cost,
                "expected_avoided_loss": avoided_loss,
                "expected_net_value": avoided_loss - cost,
            }
        )
    return pd.DataFrame(rows)


def select_operating_threshold(costs: pd.DataFrame, capacity: int) -> tuple[float, str]:
    eligible = costs[costs["customers_contacted"] <= capacity]
    pool = eligible if not eligible.empty else costs
    best = pool.sort_values(["expected_net_value", "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), (
        f"Selected {best['threshold']:.2f} using maximum simulated net value "
        f"within a contact capacity of {capacity}."
    )
