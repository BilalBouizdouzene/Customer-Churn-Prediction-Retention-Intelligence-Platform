from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_pipeline import score_frame  # noqa: E402

from churn_intelligence.config import ensure_output_dirs, load_yaml  # noqa: E402
from churn_intelligence.data.loaders import load_raw_data  # noqa: E402

if __name__ == "__main__":
    ensure_output_dirs()
    data_cfg = load_yaml("data.yaml")
    frame = load_raw_data(ROOT / data_cfg["raw_path"])
    bundle = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
    scores = score_frame(frame, bundle)
    path = ROOT / "results" / "scores" / "customer_churn_scores.csv"
    scores.to_csv(path, index=False)
    print(f"Saved {len(scores)} scores to {path}")
