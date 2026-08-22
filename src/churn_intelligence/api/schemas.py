from __future__ import annotations

from typing import Any

ALLOWED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def validate_customer_payload(payload: dict[str, Any], required_fields: list[str]) -> list[str]:
    errors: list[str] = []
    for field in required_fields:
        if field not in payload or payload[field] in (None, ""):
            errors.append(f"Champ obligatoire manquant : {field}")
    for field, allowed in ALLOWED_CATEGORIES.items():
        if field in payload and payload[field] not in allowed:
            errors.append(f"Valeur invalide pour {field}. Valeurs attendues : {sorted(allowed)}")
    for field in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if field in payload:
            try:
                value = float(payload[field])
                if value < 0:
                    errors.append(f"{field} ne peut pas être négatif")
            except (TypeError, ValueError):
                errors.append(f"{field} doit être numérique")
    if "SeniorCitizen" in payload and payload["SeniorCitizen"] not in (0, 1, "0", "1"):
        errors.append("SeniorCitizen doit valoir 0 ou 1")
    return errors
