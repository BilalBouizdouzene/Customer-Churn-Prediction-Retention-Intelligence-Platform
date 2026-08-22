from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def model_registry(config: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
    settings = config.get("models", {})
    positive = max(int(y_train.sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    xgb_settings = dict(settings.get("xgboost", {}))
    xgb_settings["scale_pos_weight"] = negative / positive
    xgb_settings.update(
        {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist"}
    )
    return {
        "logistic": LogisticRegression(
            **settings.get("logistic", {}), random_state=config.get("random_state", 42)
        ),
        "logistic_balanced": LogisticRegression(
            **settings.get("logistic_balanced", {}), random_state=config.get("random_state", 42)
        ),
        "random_forest": RandomForestClassifier(
            **settings.get("random_forest", {}), random_state=config.get("random_state", 42)
        ),
        "xgboost": XGBClassifier(**xgb_settings, random_state=config.get("random_state", 42)),
    }
