# Customer Churn Prediction & Retention Intelligence Platform

Python · Scikit-learn · XGBoost · SHAP · Flask

- Developed a leakage-safe churn classification pipeline on the public IBM Telco dataset; selected `logistic_balanced` using validation PR-AUC, recall and F1.
- Added calibrated probabilities, business-aware thresholding at `0.35`, SHAP drivers and retention-priority scoring.
- Exposed the scoring workflow through a Flask API and review dashboard; final test PR-AUC: `0.668` and recall: `0.727`.
