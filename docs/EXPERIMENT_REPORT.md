# Experiment report

## Business question
Identify telecom customers at risk of churn early enough for a retention team to prioritize an intervention.

## Dataset
IBM Telco Customer Churn, loaded from `data/raw/Telco-Customer-Churn.csv`. The executed profile contains 7043 rows and 21 columns.

## Method
The workflow uses a stratified train/validation/test split, deterministic feature engineering, and a scikit-learn `ColumnTransformer` fitted only inside the model pipeline. Numeric values are imputed and scaled; categorical values are imputed and one-hot encoded with unknown-category handling.

## Model experiments
Logistic Regression, balanced Logistic Regression, Random Forest and XGBoost were trained on the same training split. The benchmark is stored in `results/metrics/model_benchmark.csv`.

Selected model: **logistic_balanced**. Selection was based on validation PR-AUC, then recall and F1. The operating threshold is **0.35**, selected using a configurable contact-capacity and retention economics simulation.

## Final test result

| Metric | Value |
|---|---:|
| PR-AUC | 0.6678 |
| ROC-AUC | 0.8479 |
| Recall | 0.7273 |
| Precision | 0.5597 |
| F1 | 0.6326 |
| Brier score | 0.1345 |

Calibration Brier score changed from 0.1645 raw to 0.1364 calibrated on validation data.

## Explainability and decisions
Global SHAP artifacts are under `results/explainability/`. Batch scores include risk segment, priority score and a transparent heuristic recommendation. SHAP and these recommendations describe associations and decision support; they do not establish causality or guaranteed retention impact.

## Deployment and monitoring
The persisted bundle is `models/churn_pipeline.joblib`. Flask exposes `/health`, `/model/info`, `/predict`, `/predict/batch` and `/metrics`. Offline drift checks distinguish feature drift from prediction/performance monitoring; retraining is recommended only when configured signals exceed thresholds.

## Limitations
The dataset is historical and public, so it may not represent a current company. Business values are simulations. Production use requires data governance, fresh labels, calibration checks, fairness review, access controls and human oversight.
