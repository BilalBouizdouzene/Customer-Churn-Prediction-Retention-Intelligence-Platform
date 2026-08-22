from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Churn"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the public IBM Telco CSV and normalize the target/charge types."""
    frame = pd.read_csv(path)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if TARGET in frame.columns:
        if frame[TARGET].dtype == "object":
            frame[TARGET] = frame[TARGET].map({"Yes": 1, "No": 0}).astype("Int64")
        frame[TARGET] = frame[TARGET].astype(int)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in frame.columns:
        raise ValueError(f"Target column {TARGET!r} is missing")
    target = frame[TARGET].astype(int)
    features = frame.drop(columns=[TARGET]).copy()
    if "customerID" in features.columns:
        features = features.drop(columns=["customerID"])
    return features, target
