from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        metrics = classification_metrics(y_true, probabilities, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "customers_flagged": int((probabilities >= threshold).sum()),
                **{
                    key: metrics[key]
                    for key in ["precision", "recall", "f1", "false_positives", "false_negatives"]
                },
            }
        )
    return pd.DataFrame(rows)
