from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn_intelligence.config import ensure_output_dirs, load_yaml  # noqa: E402
from churn_intelligence.data.loaders import load_raw_data, split_features_target  # noqa: E402
from churn_intelligence.data.validation import (  # noqa: E402
    validate_dataset,
    write_validation_report,
)
from churn_intelligence.decisions.retention import recommendation  # noqa: E402
from churn_intelligence.decisions.segmentation import priority_score, risk_level  # noqa: E402
from churn_intelligence.evaluation.cost_analysis import (  # noqa: E402
    business_cost_table,
    select_operating_threshold,
)
from churn_intelligence.evaluation.metrics import classification_metrics  # noqa: E402
from churn_intelligence.evaluation.threshold import threshold_table  # noqa: E402
from churn_intelligence.explainability.shap_explanations import (
    save_global_explanation,  # noqa: E402
)
from churn_intelligence.features.engineering import engineer_features  # noqa: E402
from churn_intelligence.features.transformers import make_preprocessor  # noqa: E402
from churn_intelligence.models.registry import model_registry  # noqa: E402
from churn_intelligence.monitoring.drift import drift_report  # noqa: E402


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def make_pipeline(estimator: object, frame: pd.DataFrame) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor(frame)), ("model", estimator)])


def save_eda(frame: pd.DataFrame) -> None:
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    target_rate = frame.groupby("Churn", as_index=False).size()
    target_rate["Churn"] = target_rate["Churn"].map({0: "Stayed", 1: "Churned"})
    plt.figure(figsize=(6, 4))
    sns.barplot(data=target_rate, x="Churn", y="size", hue="Churn", legend=False, palette="deep")
    plt.title("Customer outcome distribution")
    plt.xlabel("")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(output / "target_distribution.png", dpi=140)
    plt.close()

    if {"tenure", "Churn"}.issubset(frame.columns):
        plt.figure(figsize=(8, 4))
        sns.histplot(
            data=frame, x="tenure", hue="Churn", bins=24, stat="density", common_norm=False
        )
        plt.title("Tenure distribution by churn outcome")
        plt.tight_layout()
        plt.savefig(output / "tenure_by_churn.png", dpi=140)
        plt.close()
    if {"MonthlyCharges", "Churn"}.issubset(frame.columns):
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=frame, x="Churn", y="MonthlyCharges", hue="Churn", legend=False)
        plt.title("Monthly charges by churn outcome")
        plt.tight_layout()
        plt.savefig(output / "monthly_charges_by_churn.png", dpi=140)
        plt.close()
    categorical = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    for column in categorical:
        if column not in frame.columns:
            continue
        rates = frame.groupby(column, observed=True)["Churn"].mean().sort_values(ascending=False)
        plt.figure(figsize=(9, 4))
        rates.plot(kind="bar", color="#0b7285")
        plt.title(f"Churn rate by {column}")
        plt.ylabel("Churn rate")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / f"churn_by_{column.lower()}.png", dpi=140)
        plt.close()


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    data_cfg = load_yaml("data.yaml")
    model_cfg = load_yaml("models.yaml")
    business_cfg = load_yaml("business.yaml")
    raw_path = ROOT / data_cfg["raw_path"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing real dataset: {raw_path}. Run scripts/download_data.py first."
        )

    frame = load_raw_data(raw_path)
    profile = validate_dataset(frame)
    write_validation_report(profile, ROOT / "results" / "reports")
    if not profile["valid"]:
        raise ValueError(f"Dataset validation failed: {profile['issues']}")
    save_eda(frame)

    indices = np.arange(len(frame))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=data_cfg["test_size"],
        stratify=frame["Churn"],
        random_state=data_cfg["random_state"],
    )
    validation_fraction = data_cfg["validation_size"] / (1 - data_cfg["test_size"])
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=validation_fraction,
        stratify=frame.iloc[train_val_idx]["Churn"],
        random_state=data_cfg["random_state"],
    )
    raw_train, raw_val, raw_test = (
        frame.iloc[train_idx].copy(),
        frame.iloc[validation_idx].copy(),
        frame.iloc[test_idx].copy(),
    )
    raw_train.to_csv(ROOT / "data" / "processed" / "train.csv", index=False)
    raw_val.to_csv(ROOT / "data" / "processed" / "validation.csv", index=False)
    raw_test.to_csv(ROOT / "data" / "processed" / "test.csv", index=False)

    def features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        x, y = split_features_target(raw)
        return engineer_features(x), y

    X_train, y_train = features(raw_train)
    X_val, y_val = features(raw_val)
    X_test, y_test = features(raw_test)
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    models = model_registry(model_cfg, y_train.to_numpy())
    benchmark_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, estimator in models.items():
        started = time.perf_counter()
        pipe = make_pipeline(estimator, X_train)
        pipe.fit(X_train, y_train)
        training_time = time.perf_counter() - started
        predict_started = time.perf_counter()
        probabilities = pipe.predict_proba(X_val)[:, 1]
        inference_ms = (time.perf_counter() - predict_started) * 1000 / max(len(X_val), 1)
        metrics = classification_metrics(y_val.to_numpy(), probabilities)
        benchmark_rows.append(
            {
                "model": name,
                "training_time_seconds": training_time,
                "inference_time_ms_per_row": inference_ms,
                **metrics,
            }
        )
        fitted[name] = pipe
        validation_probabilities[name] = probabilities
    benchmark = pd.DataFrame(benchmark_rows).sort_values(
        ["pr_auc", "recall", "f1"], ascending=False
    )
    benchmark.to_csv(ROOT / "results" / "metrics" / "model_benchmark.csv", index=False)
    dump_json(
        ROOT / "results" / "metrics" / "model_benchmark.json", benchmark.to_dict(orient="records")
    )
    selected_name = str(benchmark.iloc[0]["model"])
    selected_base = fitted[selected_name]

    calibration_model = CalibratedClassifierCV(
        estimator=selected_base, method=model_cfg.get("calibration_method", "sigmoid"), cv=3
    )
    calibration_model.fit(X_train, y_train)
    calibrated_val_prob = calibration_model.predict_proba(X_val)[:, 1]
    raw_val_prob = validation_probabilities[selected_name]
    calibration_metrics = {
        "raw_brier_score": classification_metrics(y_val.to_numpy(), raw_val_prob)["brier_score"],
        "calibrated_brier_score": classification_metrics(y_val.to_numpy(), calibrated_val_prob)[
            "brier_score"
        ],
    }
    thresholds = threshold_table(y_val.to_numpy(), calibrated_val_prob)
    costs = business_cost_table(
        y_val.to_numpy(), calibrated_val_prob, thresholds["threshold"].tolist(), business_cfg
    )
    threshold_analysis = thresholds.merge(costs, on="threshold")
    threshold_analysis.to_csv(ROOT / "results" / "metrics" / "threshold_analysis.csv", index=False)
    costs.to_csv(ROOT / "results" / "metrics" / "business_threshold_table.csv", index=False)
    selected_threshold, threshold_reason = select_operating_threshold(
        costs, int(business_cfg["team_capacity"])
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        threshold_analysis["threshold"],
        threshold_analysis["expected_net_value"],
        marker="o",
        label="Expected net value",
    )
    plt.axvline(
        selected_threshold,
        color="#e8590c",
        linestyle="--",
        label=f"Selected {selected_threshold:.2f}",
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Simulated net value")
    plt.title("Business-aware threshold analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "threshold_business_value.png", dpi=140)
    plt.close()

    fraction_pos, mean_pred = calibration_curve(
        y_val, calibrated_val_prob, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, fraction_pos, marker="o", label="Calibrated model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed churn rate")
    plt.title("Probability calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "calibration_curve.png", dpi=140)
    plt.close()

    final_model = CalibratedClassifierCV(
        estimator=make_pipeline(
            model_registry(model_cfg, y_trainval.to_numpy())[selected_name], X_trainval
        ),
        method=model_cfg.get("calibration_method", "sigmoid"),
        cv=3,
    )
    final_model.fit(X_trainval, y_trainval)
    explanation_model = make_pipeline(
        model_registry(model_cfg, y_trainval.to_numpy())[selected_name], X_trainval
    )
    explanation_model.fit(X_trainval, y_trainval)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test.to_numpy(), test_probabilities, selected_threshold)
    dump_json(ROOT / "results" / "metrics" / "final_test_metrics.json", test_metrics)

    bundle = {
        "model": final_model,
        "explanation_model": explanation_model,
        "model_name": selected_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S"),
        "feature_columns": X_train.columns.tolist(),
        "raw_feature_columns": split_features_target(raw_train)[0].columns.tolist(),
        "threshold": selected_threshold,
        "risk_thresholds": business_cfg["risk_thresholds"],
        "business_assumptions": business_cfg,
        "validation_metrics": classification_metrics(
            y_val.to_numpy(), calibrated_val_prob, selected_threshold
        ),
        "test_metrics": test_metrics,
        "calibration_metrics": calibration_metrics,
        "dataset": "IBM Telco Customer Churn",
        "dataset_churn_rate": float(frame["Churn"].mean()),
    }
    joblib.dump(bundle, ROOT / "models" / "churn_pipeline.joblib")
    dump_json(
        ROOT / "models" / "model_metadata.json",
        {key: value for key, value in bundle.items() if key not in {"model", "explanation_model"}},
    )

    importance = save_global_explanation(
        explanation_model,
        X_test.sample(min(250, len(X_test)), random_state=42),
        ROOT / "results" / "explainability",
    )
    dump_json(
        ROOT / "results" / "explainability" / "top_drivers.json",
        importance.head(15).to_dict(orient="records"),
    )

    scored = score_frame(frame, bundle)
    scored.to_csv(ROOT / "results" / "scores" / "customer_churn_scores.csv", index=False)
    false_negatives = raw_test[
        (y_test.to_numpy() == 1) & (test_probabilities < selected_threshold)
    ].copy()
    false_positives = raw_test[
        (y_test.to_numpy() == 0) & (test_probabilities >= selected_threshold)
    ].copy()
    error_report = (
        "# Error analysis\n\n"
        f"Operating threshold: `{selected_threshold:.2f}`\n\n"
        f"- Test false negatives: {len(false_negatives)}\n"
        f"- Test false positives: {len(false_positives)}\n\n"
        "## Interpretation\n\n"
        "False negatives are churners missed by the intervention queue; false positives are customers contacted who did not churn. "
        "Use the saved test rows for segment-level follow-up. These associations are diagnostic, not causal.\n"
    )
    write_report(ROOT / "results" / "reports" / "error_analysis.md", error_report)
    write_report(
        ROOT / "results" / "reports" / "threshold_selection.md",
        (
            "# Threshold selection\n\n"
            f"{threshold_reason}\n\n"
            "The economic values are configurable simulation assumptions from `configs/business.yaml`, not observed company economics.\n"
        ),
    )
    write_report(
        ROOT / "results" / "reports" / "model_selection.md",
        (
            "# Model selection\n\n"
            f"Selected model: **{selected_name}**. The selection maximized validation PR-AUC, then recall and F1.\n\n"
            f"Calibration comparison: raw Brier score `{calibration_metrics['raw_brier_score']:.4f}`, calibrated Brier score `{calibration_metrics['calibrated_brier_score']:.4f}`.\n\n"
            "The final test set was kept outside model selection and is reported in `results/metrics/final_test_metrics.json`.\n"
        ),
    )
    dump_json(ROOT / "results" / "reports" / "drift_report.json", drift_report(X_train, X_test))
    write_experiment_report(
        selected_name, selected_threshold, benchmark, test_metrics, calibration_metrics, profile
    )
    write_report(
        ROOT / "docs" / "CV_PROJECT_SUMMARY.md",
        (
            "# Customer Churn Prediction & Retention Intelligence Platform\n\n"
            "Python · Scikit-learn · XGBoost · SHAP · Flask\n\n"
            f"- Developed a leakage-safe churn classification pipeline on the public IBM Telco dataset; selected `{selected_name}` using validation PR-AUC, recall and F1.\n"
            f"- Added calibrated probabilities, business-aware thresholding at `{selected_threshold:.2f}`, SHAP drivers and retention-priority scoring.\n"
            f"- Exposed the scoring workflow through a Flask API and review dashboard; final test PR-AUC: `{test_metrics['pr_auc']:.3f}` and recall: `{test_metrics['recall']:.3f}`.\n"
        ),
    )
    print(
        json.dumps(
            {
                "selected_model": selected_name,
                "threshold": selected_threshold,
                "test_metrics": test_metrics,
            },
            indent=2,
        )
    )


def score_frame(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features, _ = split_features_target(frame)
    features = engineer_features(features)
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    rows = []
    thresholds = bundle["risk_thresholds"]
    for index, probability in enumerate(probabilities):
        original = frame.iloc[index]
        rows.append(
            {
                "customer_id": original.get("customerID", str(index)),
                "churn_probability": round(float(probability), 6),
                "risk_level": risk_level(float(probability), thresholds),
                "priority_score": priority_score(
                    float(probability),
                    original.get("MonthlyCharges"),
                    bundle["business_assumptions"],
                ),
                "recommended_action": recommendation(original, float(probability)),
                "model_version": bundle["model_version"],
            }
        )
    return pd.DataFrame(rows).sort_values(["priority_score", "churn_probability"], ascending=False)


def write_experiment_report(
    selected_name: str,
    threshold: float,
    benchmark: pd.DataFrame,
    test_metrics: dict,
    calibration: dict,
    profile: dict,
) -> None:
    report = f"""# Experiment report

## Business question
Identify telecom customers at risk of churn early enough for a retention team to prioritize an intervention.

## Dataset
IBM Telco Customer Churn, loaded from `data/raw/Telco-Customer-Churn.csv`. The executed profile contains {profile["rows"]} rows and {profile["columns"]} columns.

## Method
The workflow uses a stratified train/validation/test split, deterministic feature engineering, and a scikit-learn `ColumnTransformer` fitted only inside the model pipeline. Numeric values are imputed and scaled; categorical values are imputed and one-hot encoded with unknown-category handling.

## Model experiments
Logistic Regression, balanced Logistic Regression, Random Forest and XGBoost were trained on the same training split. The benchmark is stored in `results/metrics/model_benchmark.csv`.

Selected model: **{selected_name}**. Selection was based on validation PR-AUC, then recall and F1. The operating threshold is **{threshold:.2f}**, selected using a configurable contact-capacity and retention economics simulation.

## Final test result

| Metric | Value |
|---|---:|
| PR-AUC | {test_metrics["pr_auc"]:.4f} |
| ROC-AUC | {test_metrics["roc_auc"]:.4f} |
| Recall | {test_metrics["recall"]:.4f} |
| Precision | {test_metrics["precision"]:.4f} |
| F1 | {test_metrics["f1"]:.4f} |
| Brier score | {test_metrics["brier_score"]:.4f} |

Calibration Brier score changed from {calibration["raw_brier_score"]:.4f} raw to {calibration["calibrated_brier_score"]:.4f} calibrated on validation data.

## Explainability and decisions
Global SHAP artifacts are under `results/explainability/`. Batch scores include risk segment, priority score and a transparent heuristic recommendation. SHAP and these recommendations describe associations and decision support; they do not establish causality or guaranteed retention impact.

## Deployment and monitoring
The persisted bundle is `models/churn_pipeline.joblib`. Flask exposes `/health`, `/model/info`, `/predict`, `/predict/batch` and `/metrics`. Offline drift checks distinguish feature drift from prediction/performance monitoring; retraining is recommended only when configured signals exceed thresholds.

## Limitations
The dataset is historical and public, so it may not represent a current company. Business values are simulations. Production use requires data governance, fresh labels, calibration checks, fairness review, access controls and human oversight.
"""
    write_report(ROOT / "docs" / "EXPERIMENT_REPORT.md", report)


if __name__ == "__main__":
    main()
