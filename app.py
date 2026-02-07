import os
import json
import pickle
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------
# CONFIG (edit these 2 things)
# ---------------------------

# 1) Your model file name in root folder (same folder as app.py)
MODEL_FILE = "model.pkl"   # change if your file name differs, e.g. "silver_model.pkl"

# 2) Feature names used during training (MUST match training order)
# If you don't know, check your training notebook where you did: X = df[[...]] or feature_cols = [...]
FEATURE_NAMES = [
    # Example only. REPLACE with your real feature list.
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "RSI_14"
]

# Optional: if you have meta.json with {"feature_names":[...]} it will override FEATURE_NAMES
META_FILE = "meta.json"


# ---------------------------
# Model Loading (Railway-safe)
# ---------------------------
def load_model_and_feature_names():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILE)
    meta_path = os.path.join(base_dir, META_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    feature_names = FEATURE_NAMES

    # If meta.json exists, use it (recommended)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict) and meta.get("feature_names"):
                feature_names = meta["feature_names"]
        except Exception as e:
            print("⚠️ meta.json exists but couldn't be read. Using FEATURE_NAMES from app.py")
            print("meta.json error:", e)

    if not feature_names or not isinstance(feature_names, list):
        raise ValueError("❌ feature_names is empty/invalid. Set FEATURE_NAMES or provide meta.json.")

    return model, feature_names


MODEL, FEATURE_COLS = load_model_and_feature_names()


# -----------------------------------
# Helper: build next-day features safely
# -----------------------------------
def build_next_day_features_safe(payload):
    """
    Supports two inputs:
    1) payload["last_rows"] = list of dicts (history rows)
    2) payload["features_dict"] OR payload["features"] for direct prediction

    Returns:
        X_next (np.ndarray shape (1, n_features))
    """
    last_rows = payload.get("last_rows")
    features_dict = payload.get("features_dict")
    features_list = payload.get("features")

    # Case 1: History provided
    if last_rows is not None:
        df = pd.DataFrame(last_rows)
        if df.empty:
            raise ValueError("last_rows is empty; cannot compute next-day features.")

        # Ensure required columns exist
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in last_rows: {missing}")

        # Use last row as base
        base = df.iloc[-1].copy()

        # Convert numeric columns safely
        for c in FEATURE_COLS:
            base[c] = pd.to_numeric(base[c], errors="coerce")

        # ---- SAFE rolling/technical indicators (only if needed)
        # If your model uses these features, calculate them safely from Close
        if "Close" in df.columns:
            close = pd.to_numeric(df["Close"], errors="coerce")

            if "MA_7" in FEATURE_COLS:
                base["MA_7"] = close.rolling(7, min_periods=1).mean().iloc[-1]
            if "MA_14" in FEATURE_COLS:
                base["MA_14"] = close.rolling(14, min_periods=1).mean().iloc[-1]
            if "RSI_14" in FEATURE_COLS:
                delta = close.diff()
                gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
                loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
                rs = gain / (loss.replace(0, np.nan))
                rsi = 100 - (100 / (1 + rs))
                base["RSI_14"] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

        # Build row in correct order
        row = []
        for col in FEATURE_COLS:
            val = base.get(col, np.nan)
            row.append(val)

        X_next = np.array(row, dtype=float).reshape(1, -1)
        X_next = np.nan_to_num(X_next, nan=0.0, posinf=0.0, neginf=0.0)

        return X_next

    # Case 2: Direct dict
    if features_dict is not None:
        row = []
        for col in FEATURE_COLS:
            if col not in features_dict:
                raise ValueError(f"Missing feature '{col}' in features_dict")
            row.append(float(features_dict[col]))
        X = np.array(row, dtype=float).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # Case 3: Direct list
    if features_list is not None:
        X = np.array(features_list, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(FEATURE_COLS):
            raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_COLS)} got {X.shape[1]}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    raise ValueError("Send one of: last_rows OR features_dict OR features")


# ---------------------------
# Pages
# ---------------------------
@app.get("/")
def home():
    # keep your existing template name, change if yours is different
    return render_template("index.html")


# ---------------------------
# API: Predict (today)
# ---------------------------
@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}

        # support both list and dict
        features_dict = payload.get("features_dict")
        features_list = payload.get("features")

        if features_dict is None and features_list is None:
            raise ValueError("Missing input: send 'features' (array) OR 'features_dict' (object).")

        if features_dict is not None:
            row = []
            for col in FEATURE_COLS:
                if col not in features_dict:
                    raise ValueError(f"Missing feature '{col}' in features_dict.")
                row.append(float(features_dict[col]))
            X = np.array(row, dtype=float).reshape(1, -1)
        else:
            X = np.array(features_list, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        if X.shape[1] != len(FEATURE_COLS):
            raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_COLS)}, got {X.shape[1]}")

        if not np.isfinite(X).all():
            raise ValueError("Inputs contain NaN/inf. Please fill all fields.")

        pred = MODEL.predict(X)
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

        return jsonify({"success": True, "prediction": pred_value})

    except Exception as e:
        print("❌ /predict error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction.please try again",
            "debug_error": str(e)  # remove later
        }), 500


# ---------------------------
# API: Predict Next Day
# ---------------------------
@app.post("/predict-next-day")
def predict_next_day():
    try:
        payload = request.get_json(silent=True) or {}

        X_next = build_next_day_features_safe(payload)

        if X_next.shape[1] != len(FEATURE_COLS):
            raise ValueError(f"Next-day feature mismatch. Expected {len(FEATURE_COLS)}, got {X_next.shape[1]}")

        if not np.isfinite(X_next).all():
            raise ValueError("Next-day features contain NaN/inf (rolling/lag issue).")

        pred = MODEL.predict(X_next)
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        return jsonify({"success": True, "date": tomorrow, "prediction": pred_value})

    except Exception as e:
        print("❌ /predict-next-day error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something wen wrong while calculating prediction.please try again",
            "debug_error": str(e)  # keep while debugging
        }), 500


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
