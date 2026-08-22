import pandas as pd

from churn_intelligence.data.validation import validate_dataset
from churn_intelligence.features.engineering import engineer_features


def test_total_charges_and_validation():
    frame = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [0, 3],
            "MonthlyCharges": [20.0, 30.0],
            "TotalCharges": ["", "90"],
            "Churn": [0, 1],
        }
    )
    profile = validate_dataset(frame)
    assert profile["rows"] == 2
    assert profile["valid"] is True
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    assert frame["TotalCharges"].isna().sum() == 1


def test_feature_engineering_has_no_target_column():
    frame = pd.DataFrame(
        {
            "tenure": [1, 20],
            "MonthlyCharges": [90.0, 40.0],
            "TotalCharges": [90.0, 800.0],
            "Contract": ["Month-to-month", "Two year"],
            "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
            "InternetService": ["Fiber optic", "DSL"],
            "PhoneService": ["Yes", "No"],
            "TechSupport": ["No", "Yes"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "Yes"],
        }
    )
    result = engineer_features(frame)
    assert "Churn" not in result.columns
    assert result.loc[0, "long_term_contract"] == 0
    assert result.loc[1, "auto_payment"] == 1
    assert result.loc[0, "services_count"] > 0
