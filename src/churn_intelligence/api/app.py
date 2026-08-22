from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

from churn_intelligence.api.schemas import validate_customer_payload
from churn_intelligence.config import ROOT, load_yaml
from churn_intelligence.data.loaders import load_raw_data
from churn_intelligence.decisions.retention import recommendation
from churn_intelligence.decisions.segmentation import priority_score, risk_level
from churn_intelligence.explainability.shap_explanations import local_explanation
from churn_intelligence.features.engineering import engineer_features


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "frontend" / "templates"),
        static_folder=str(ROOT / "frontend" / "static"),
    )
    bundle_path = ROOT / "models" / "churn_pipeline.joblib"
    bundle = joblib.load(bundle_path) if bundle_path.exists() else None
    if test_config:
        app.config.update(test_config)

    def ensure_bundle() -> dict[str, Any]:
        nonlocal bundle
        if bundle is None:
            raise RuntimeError("Modèle absent. Exécutez python scripts/run_pipeline.py")
        return bundle

    def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
        model_bundle = ensure_bundle()
        errors = validate_customer_payload(payload, model_bundle["raw_feature_columns"])
        if errors:
            raise ValueError("; ".join(errors))
        raw = pd.DataFrame([{key: payload.get(key) for key in model_bundle["raw_feature_columns"]}])
        if "SeniorCitizen" in raw:
            raw["SeniorCitizen"] = pd.to_numeric(raw["SeniorCitizen"], errors="coerce")
        features = engineer_features(raw)
        features = features.reindex(columns=model_bundle["feature_columns"])
        probability = float(model_bundle["model"].predict_proba(features)[:, 1][0])
        explanation = {"top_positive_drivers": [], "top_negative_drivers": []}
        try:
            explanation = local_explanation(model_bundle["explanation_model"], features)
        except Exception:
            pass
        original = raw.iloc[0]
        return {
            "churn_probability": round(probability, 6),
            "risk_level": risk_level(probability, model_bundle["risk_thresholds"]),
            "retention_priority": priority_score(
                probability,
                float(original.get("MonthlyCharges", 0)),
                model_bundle["business_assumptions"],
            ),
            "top_risk_drivers": explanation["top_positive_drivers"],
            "protective_factors": explanation["top_negative_drivers"],
            "recommended_action": recommendation(original, probability),
            "model_version": model_bundle["model_version"],
            "threshold": model_bundle["threshold"],
        }

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "model_loaded": bundle is not None})

    @app.get("/model/info")
    def model_info() -> Any:
        model_bundle = ensure_bundle()
        return jsonify(
            {
                key: model_bundle[key]
                for key in [
                    "model_name",
                    "model_version",
                    "threshold",
                    "feature_columns",
                    "raw_feature_columns",
                    "validation_metrics",
                    "test_metrics",
                    "calibration_metrics",
                    "dataset",
                ]
            }
        )

    @app.post("/predict")
    def predict() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Le corps doit être un objet JSON."}), 400
        try:
            return jsonify(predict_one(payload))
        except (RuntimeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict/batch")
    def predict_batch() -> Any:
        payload = request.get_json(silent=True)
        customers = payload.get("customers") if isinstance(payload, dict) else payload
        if not isinstance(customers, list) or not customers:
            return jsonify({"error": "Envoyez une liste de clients ou {'customers': [...]}"}), 400
        results, errors = [], []
        for index, customer in enumerate(customers):
            try:
                results.append(predict_one(customer))
            except (RuntimeError, ValueError) as exc:
                errors.append({"index": index, "error": str(exc)})
        return jsonify({"predictions": results, "errors": errors})

    @app.get("/metrics")
    def metrics() -> Any:
        model_bundle = ensure_bundle()
        benchmark_path = ROOT / "results" / "metrics" / "model_benchmark.csv"
        benchmark = (
            pd.read_csv(benchmark_path).to_dict(orient="records") if benchmark_path.exists() else []
        )
        return jsonify({"final_test": model_bundle["test_metrics"], "benchmark": benchmark})

    @app.get("/api/overview")
    def overview() -> Any:
        scores = _load_scores()
        return jsonify(_overview(scores))

    @app.get("/api/scores")
    def scores_api() -> Any:
        scores = _load_scores()
        risk = request.args.get("risk")
        contract = request.args.get("contract")
        internet = request.args.get("internet")
        if risk:
            scores = scores[scores["risk_level"].str.upper() == risk.upper()]
        if contract and "Contract" in scores:
            scores = scores[scores["Contract"] == contract]
        if internet and "InternetService" in scores:
            scores = scores[scores["InternetService"] == internet]
        limit = min(max(int(request.args.get("limit", 500)), 1), 10000)
        return jsonify(scores.head(limit).to_dict(orient="records"))

    @app.get("/api/customer/<customer_id>")
    def customer_api(customer_id: str) -> Any:
        raw_path = ROOT / load_yaml("data.yaml")["raw_path"]
        if not raw_path.exists():
            return jsonify({"error": "Dataset brut absent."}), 404
        frame = load_raw_data(raw_path)
        customer = frame[frame["customerID"].astype(str) == customer_id]
        if customer.empty:
            return jsonify({"error": "Client introuvable."}), 404
        values = customer.iloc[0].drop(labels=["Churn"], errors="ignore").where(lambda item: item.notna(), None).to_dict()
        safe_values = json.loads(json.dumps(values, default=lambda value: value.item() if hasattr(value, "item") else None))
        scores = _load_scores()
        score = scores[scores["customer_id"].astype(str) == customer_id]
        score_values = score.iloc[0].to_dict() if not score.empty else {}
        return jsonify({"customer": safe_values, "score": score_values})

    @app.get("/assets/<path:asset_path>")
    def result_asset(asset_path: str) -> Any:
        result_root = (ROOT / "results").resolve()
        candidate = (result_root / asset_path).resolve()
        if result_root not in candidate.parents or not candidate.is_file():
            abort(404)
        return send_file(candidate)

    @app.get("/")
    def index() -> Any:
        scores = _load_scores()
        benchmark = _load_csv("results/metrics/model_benchmark.csv")
        threshold_analysis = _load_csv("results/metrics/threshold_analysis.csv")
        drivers = _load_csv("results/explainability/global_feature_importance.csv")
        return render_template(
            "index.html",
            overview=_overview(scores),
            top_scores=scores.head(12).to_dict(orient="records"),
            benchmark=benchmark.to_dict(orient="records"),
            threshold_analysis=threshold_analysis.head(8).to_dict(orient="records"),
            drivers=drivers.head(12).to_dict(orient="records"),
            model=ensure_bundle(),
        )

    def _load_scores() -> pd.DataFrame:
        path = ROOT / "results" / "scores" / "customer_churn_scores.csv"
        scores = (
            pd.read_csv(path)
            if path.exists()
            else pd.DataFrame(
                columns=[
                    "customer_id",
                    "churn_probability",
                    "risk_level",
                    "priority_score",
                    "recommended_action",
                ]
            )
        )
        raw_path = ROOT / load_yaml("data.yaml")["raw_path"]
        if raw_path.exists() and not scores.empty:
            raw = load_raw_data(raw_path)
            fields = raw[["customerID", "Contract", "InternetService"]].rename(
                columns={"customerID": "customer_id"}
            )
            scores = scores.merge(fields, on="customer_id", how="left")
        return scores

    def _load_csv(relative_path: str) -> pd.DataFrame:
        path = ROOT / relative_path
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _overview(scores: pd.DataFrame) -> dict[str, Any]:
        model_bundle = ensure_bundle()
        return {
            "total_customers": int(len(scores)),
            "churn_rate": float(model_bundle.get("dataset_churn_rate", 0)),
            "high_risk_customers": int(scores["risk_level"].isin(["HIGH", "CRITICAL"]).sum())
            if not scores.empty
            else 0,
            "critical_risk_customers": int((scores["risk_level"] == "CRITICAL").sum())
            if not scores.empty
            else 0,
            "recommended_contacts": int((scores["risk_level"].isin(["HIGH", "CRITICAL"])).sum())
            if not scores.empty
            else 0,
            "expected_retained_value": round(
                float(
                    scores.loc[
                        scores["risk_level"].isin(["HIGH", "CRITICAL"]), "priority_score"
                    ].sum()
                ),
                2,
            )
            if not scores.empty
            else 0.0,
        }

    return app


app = create_app()


if __name__ == "__main__":
    config = load_yaml("app.yaml")
    app.run(
        host=config.get("host", "127.0.0.1"),
        port=int(config.get("port", 5000)),
        debug=bool(config.get("debug", False)),
    )
