# Model selection

Selected model: **logistic_balanced**. The selection maximized validation PR-AUC, then recall and F1.

Calibration comparison: raw Brier score `0.1645`, calibrated Brier score `0.1364`.

The final test set was kept outside model selection and is reported in `results/metrics/final_test_metrics.json`.
