from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def validate_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    issues: list[str] = []
    if frame.empty:
        issues.append("Dataset is empty")
    if "Churn" not in frame.columns:
        issues.append("Missing target column: Churn")
    if "customerID" in frame.columns and frame["customerID"].duplicated().any():
        issues.append("Duplicate customer IDs detected")
    if "tenure" in frame.columns and (pd.to_numeric(frame["tenure"], errors="coerce") < 0).any():
        issues.append("Negative tenure detected")
    for column in ["MonthlyCharges", "TotalCharges"]:
        if column in frame.columns and (pd.to_numeric(frame[column], errors="coerce") < 0).any():
            issues.append(f"Negative values detected in {column}")
    if "Churn" in frame.columns and not set(frame["Churn"].dropna().unique()).issubset(
        {0, 1, "Yes", "No"}
    ):
        issues.append("Target is not binary")

    missing = frame.isna().sum().sort_values(ascending=False)
    schema_missing = sorted(EXPECTED_COLUMNS - set(frame.columns))
    profile = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": frame.columns.tolist(),
        "missing_columns_from_expected_schema": schema_missing,
        "duplicate_customer_ids": int(frame["customerID"].duplicated().sum())
        if "customerID" in frame
        else None,
        "missing_values": {k: int(v) for k, v in missing.items() if v},
        "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
        "target_distribution": frame["Churn"].value_counts(dropna=False).to_dict()
        if "Churn" in frame
        else {},
        "issues": issues,
        "valid": not issues,
    }
    return profile


def write_validation_report(profile: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "data_profile.json").open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, default=str)
    lines = [
        "# Data validation report",
        "",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['columns']}",
        f"- Valid: `{profile['valid']}`",
        f"- Duplicate customer IDs: {profile['duplicate_customer_ids']}",
        "",
        "## Issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in profile["issues"] or ["None detected"])
    lines.extend(["", "## Missing values", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in profile["missing_values"].items())
    (output_path / "data_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
