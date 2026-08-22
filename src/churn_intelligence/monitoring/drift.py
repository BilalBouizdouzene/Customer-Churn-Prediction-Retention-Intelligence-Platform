from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference.astype(float), errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current.astype(float), errors="coerce").dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(current, bins=edges)
    expected = np.clip(expected / max(expected.sum(), 1), 1e-6, None)
    actual = np.clip(actual / max(actual.sum(), 1), 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for column in reference.columns.intersection(current.columns):
        if pd.api.types.is_numeric_dtype(reference[column]):
            ref = pd.to_numeric(reference[column].astype(float), errors="coerce").dropna()
            cur = pd.to_numeric(current[column].astype(float), errors="coerce").dropna()
            statistic, p_value = ks_2samp(ref, cur) if len(ref) and len(cur) else (0.0, 1.0)
            rows.append(
                {
                    "feature": column,
                    "kind": "numeric",
                    "psi": psi(ref, cur),
                    "ks_p_value": float(p_value),
                }
            )
        else:
            ref_dist = reference[column].fillna("<NA>").value_counts(normalize=True)
            cur_dist = current[column].fillna("<NA>").value_counts(normalize=True)
            categories = set(ref_dist.index) | set(cur_dist.index)
            distance = (
                sum(
                    abs(float(ref_dist.get(cat, 0)) - float(cur_dist.get(cat, 0)))
                    for cat in categories
                )
                / 2
            )
            rows.append(
                {"feature": column, "kind": "categorical", "distribution_distance": float(distance)}
            )
    return {
        "features": rows,
        "retraining_recommended": any(row.get("psi", 0) > 0.20 for row in rows),
    }
