import os
import traceback
from datetime import datetime, timedelta

import numpy as np
from flask import Flask, render_template, request, jsonify

from model_utils import load_or_create_demo_model, FEATURE_NAMES

app = Flask(__name__)

# Load model safely (works on Railway too)
model = load_or_create_demo_model()

# Simple in-memory history (kept during runtime)
# In real apps, store in DB / file
PREDICTION_HISTORY = []


@app.get("/")
def home():
    return render_template("index.html", feature_names=FEATURE_NAMES)


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/reviews")
def reviews():
    return render_template("reviews.html")


@app.post("/api/predict")
def api_predict():
    """
    Predict based on input features (same-day prediction)
    """
    try:
        payload = request.get_json(silent=True) or {}
        features = payload.get("features")

        if features is None:
            raise ValueError("Missing 'features' in request JSON. Expected: {features: [...]}")

        X = np.array(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Validate shape
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_NAMES)} but got {X.shape[1]}")

        # Validate values
        if not np.isfinite(X).all():
            raise ValueError("features contain NaN/inf. Please fill all inputs correctly.")

        pred = model.predict(X)
        pred_value = float(pred[0])

        result_label = "Approved ✅" if pred_value == 1.0 else "Rejected ❌"

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features": features,
            "result": result_label
        }
        PREDICTION_HISTORY.append(record)
        PREDICTION_HISTORY[:] = PREDICTION_HISTORY[-50:]  # keep last 50

        return jsonify({"success": True, "prediction": pred_value, "label": result_label})

    except Exception as e:
        print("❌ /api/predict failed:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. Please try again.",
            "debug_error": str(e)  # remove later if you want
        }), 500


@app.post("/api/predict-next-day")
def api_predict_next_day():
    """
    Next-day prediction:
    - We simulate "next day" by taking last inputs and applying a small deterministic drift,
      then predicting again.
    - The main goal: robust validation + no crashes.
    """
    try:
        payload = request.get_json(silent=True) or {}
        features = payload.get("features")

        if features is None:
            raise ValueError("Missing 'features' in request JSON. Expected: {features: [...]}")

        X = np.array(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_NAMES)} but got {X.shape[1]}")

        if not np.isfinite(X).all():
            raise ValueError("features contain NaN/inf. Please fill all inputs correctly.")

        # ---- Next-day feature drift (safe + simple)
        X_next = X.copy()

        # Example drift:
        # - Credit score: +/- 1
        # - DTI: +/- 0.2
        # - Income: +0.5% (small increase)
        # - Loan amount: unchanged
        # - Employment length: unchanged
        # - Age: unchanged
        # Adjust indices based on FEATURE_NAMES order in model_utils.py
        def idx(name): return FEATURE_NAMES.index(name)

        X_next[0, idx("income")] = X_next[0, idx("income")] * 1*
