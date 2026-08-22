from __future__ import annotations

import pandas as pd


def recommendation(row: pd.Series, probability: float) -> str:
    contract = str(row.get("Contract", ""))
    internet = str(row.get("InternetService", ""))
    support = str(row.get("TechSupport", ""))
    monthly = float(row.get("MonthlyCharges", 0) or 0)
    if probability < 0.50:
        return "Surveiller le client et privilégier une communication de fidélisation."
    if contract == "Month-to-month":
        return "Proposer une offre de passage à un contrat long terme."
    if monthly >= 80:
        return "Revoir le prix ou proposer une remise personnalisée."
    if internet != "No" and support in {"No", "No internet service"}:
        return "Proposer une option d'assistance technique ou un accompagnement dédié."
    return "Contacter le client pour comprendre les irritants et proposer une offre adaptée."
