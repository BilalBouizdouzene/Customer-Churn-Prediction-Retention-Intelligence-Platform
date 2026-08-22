from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SUPPORT_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def _yes(value: object) -> int:
    return int(str(value).strip().lower() == "yes")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features using only customer attributes available at scoring time."""
    result = frame.copy()
    if "TotalCharges" in result.columns:
        result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    if "tenure" in result.columns:
        tenure = pd.to_numeric(result["tenure"], errors="coerce").fillna(0)
        result["tenure_bucket"] = pd.cut(
            tenure,
            bins=[-1, 6, 12, 24, 48, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"],
        ).astype(str)
    available_services = [col for col in SERVICE_COLUMNS if col in result.columns]
    if available_services:
        result["services_count"] = result[available_services].apply(
            lambda row: sum(_yes(value) for value in row), axis=1
        )
    available_support = [col for col in SUPPORT_COLUMNS if col in result.columns]
    if available_support:
        result["support_services_count"] = result[available_support].apply(
            lambda row: sum(_yes(value) for value in row), axis=1
        )
        result["has_security_bundle"] = (
            result[available_support].apply(
                lambda row: all(_yes(value) for value in row[:2]), axis=1
            )
            if len(available_support) >= 2
            else 0
        )
    if "Contract" in result.columns:
        result["long_term_contract"] = result["Contract"].isin(["One year", "Two year"]).astype(int)
    if "PaymentMethod" in result.columns:
        result["auto_payment"] = (
            result["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
        )
    if {"MonthlyCharges", "TotalCharges"}.issubset(result.columns):
        monthly = pd.to_numeric(result["MonthlyCharges"], errors="coerce")
        total = pd.to_numeric(result["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(result.get("tenure", 0), errors="coerce").fillna(0)
        result["avg_monthly_value"] = total / tenure.replace(0, np.nan)
        result["charge_per_tenure"] = total / (tenure + 1)
        result["avg_monthly_value"] = result["avg_monthly_value"].replace([np.inf, -np.inf], np.nan)
        result["charge_per_tenure"] = result["charge_per_tenure"].replace([np.inf, -np.inf], np.nan)
        result["monthly_to_total_ratio"] = monthly / (total + 1)
    return result
