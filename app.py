"""
app.py — Flask web application for Customer Churn Prediction.
Serves a UI form and exposes a /predict REST API endpoint.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

# ── Load artifacts ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(path):
    full_path = os.path.join(BASE_DIR, path)
    with open(full_path, "rb") as f:
        return pickle.load(f)

model          = load_artifact("artifacts/model.pkl")
scaler         = load_artifact("artifacts/scaler.pkl")
label_encoders = load_artifact("artifacts/label_encoders.pkl")

with open(os.path.join(BASE_DIR, "artifacts/evaluation_report.json")) as f:
    eval_report = json.load(f)

# ── Feature order (must match training) ─────────────────────────
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

def preprocess_input(form_data: dict) -> np.ndarray:
    """Transform raw form input into model-ready array."""
    df = pd.DataFrame([form_data])

    # Encode categoricals using saved label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].astype(str).values[0]
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                df[col] = le.transform([le.classes_[0]])

    # Ensure correct numeric types
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[FEATURE_ORDER].astype(float)

    # Scale all numeric features using the fitted scaler
    # The scaler was fit on all 19 feature columns in the same FEATURE_ORDER
    df_scaled = scaler.transform(df)
    return df_scaled


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert numeric fields
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            data[col] = float(data.get(col, 0))
        data["SeniorCitizen"] = int(data.get("SeniorCitizen", 0))

        features = preprocess_input(data)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        result = {
            "prediction": prediction,
            "label": "Will Churn" if prediction == 1 else "Will Not Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": (
                "High Risk" if probability >= 0.7
                else "Medium Risk" if probability >= 0.4
                else "Low Risk"
            )
        }
        return render_template("result.html", result=result, input_data=data)

    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint — accepts JSON, returns JSON prediction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            data[col] = float(data.get(col, 0))
        data["SeniorCitizen"] = int(data.get("SeniorCitizen", 0))

        features = preprocess_input(data)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        return jsonify({
            "prediction": prediction,
            "label": "Will Churn" if prediction == 1 else "Will Not Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": (
                "High Risk" if probability >= 0.7
                else "Medium Risk" if probability >= 0.4
                else "Low Risk"
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return render_template("metrics.html", report=eval_report)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": eval_report.get("model_name", "unknown")})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
