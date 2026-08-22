from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _transformed(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    values = preprocessor.transform(frame)
    names = preprocessor.get_feature_names_out().tolist()
    return np.asarray(values), names


def shap_values(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    transformed, names = _transformed(model, frame)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "get_booster") or estimator.__class__.__name__ in {
        "RandomForestClassifier",
        "ExtraTreesClassifier",
    }:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[-1]
    else:
        # A zero baseline is stable for standardized numeric and one-hot features.
        # Using the single customer as background would make local contributions zero.
        background = np.zeros((1, transformed.shape[1]))
        explainer = shap.LinearExplainer(estimator, background)
        values = explainer.shap_values(transformed)
        base = explainer.expected_value
    return np.asarray(values), transformed, names


def save_global_explanation(
    model: Any, frame: pd.DataFrame, output_dir: str | Path
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values, transformed, names = shap_values(model, frame)
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output / "global_feature_importance.csv", index=False)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()
    return importance


def local_explanation(model: Any, frame: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    values, _, names = shap_values(model, frame.iloc[[0]])
    contributions = pd.DataFrame({"feature": names, "shap_value": values[0]})
    positive = contributions.sort_values("shap_value", ascending=False).head(top_n)
    negative = contributions.sort_values("shap_value", ascending=True).head(top_n)
    return {
        "top_positive_drivers": [
            {"feature": str(row.feature), "shap_value": round(float(row.shap_value), 5)}
            for row in positive.itertuples()
            if row.shap_value > 0
        ],
        "top_negative_drivers": [
            {"feature": str(row.feature), "shap_value": round(float(row.shap_value), 5)}
            for row in negative.itertuples()
            if row.shap_value < 0
        ],
    }
