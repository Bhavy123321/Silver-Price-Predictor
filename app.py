import os
import pickle
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# -----------------------------
# BASIC CONFIG (EDIT THESE 3)
# -----------------------------

# 1) Put your actual model file name here (same folder as this app.py)
MODEL_FILE = "model.pkl"  # e.g. "silver_model.pkl" / "rf_model.pkl" / "best_model.pkl"

# 2) Your main template file name (keep as-is if your template is index.html)
TEMPLATE_NAME = "index.html"  # e.g. "home.html"

# 3) Feature columns used during training (MUST match order used in training)
# Example list below. Replace with your real training feature list.
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "RSI_14"
]


# -----------------------------
# APP INIT
# -----------------------------
app = Flask(__name__)


# -----------------------------
# LOAD MODEL (Railway safe)
# -----------------------------
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()


# -----------------------------
# HELPERS
# -----------------------------
def _to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def build_features_from_dict(features_dict: dict) -> np.ndarray:
    """
    Build a (1, n_features) numpy array in exact FEATURE_COLS order.
    """
    if not isinstance(features_dict, dict):
        raise ValueError("features_dict must be an object/dict")

    row = []
    for col in FEATURE_COLS:
        if col not in features_dict:
            raise ValueError(f"Missing feature '{col}' in features_dict")
        row.append(_to_float(features_dict[col]))

    X = np.array(row, dtype=float).reshape(1, -1)

    if X.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_COLS)} got {X.shape[1]}")

    if not np.isfinite(X).all():
        raise ValueError("Input features contain NaN/inf. Please fill all inputs correctly.")

    return X


def build_features_from_list(features_list: list) -> np.ndarray:
    """
    Build a (1, n_features) numpy array from list, validate shape.
    """
    if not isinstance(features_list, list):
        raise ValueError("features must be an array/list")

    X = np.array(features_list, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_COLS)} got {X.shape[1]}")

    if not np.isfinite(X).all():
        raise ValueError("Input features contain NaN/inf. Please fill all inputs correctly.")

    return X


def compute_next_day_X_from_history(last_rows: list) -> np.ndarray:
    """
    Build next-day features safely from provided history rows.
    Expects last_rows = list of dicts (each row has at least Open/High/Low/Close/Volume)
    Creates MA_7, MA_14, RSI_14 safely if those exist in FEATURE_COLS.

    Returns:
        X_next (1, n_features)
    """
    if not last_rows or not isinstance(last_rows, list):
        raise ValueError("last_rows is missing or not a list")

    df = pd.DataFrame(last_rows)
    if df.empty:
        raise ValueError("History dataframe is empty")

    # Ensure required base columns exist when you compute indicators
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in base_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # If your model requires MA/RSI, compute them safely (no NaN bombs)
    if "Close" in df.columns:
        close = pd.to_numeric(df["Close"], errors="coerce")

        if "MA_7" in FEATURE_COLS:
            df["MA_7"] = close.rolling(7, min_periods=1).mean()

        if "MA_14" in FEATURE_COLS:
            df["MA_14"] = close.rolling(14, min_periods=1).mean()

        if "RSI_14" in FEATURE_COLS:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()

            # Avoid divide-by-zero -> NaN
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            df["RSI_14"] = rsi.fillna(50)

    # Ensure all needed feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for next-day prediction: {missing}. "
            f"Send them in last_rows OR remove them from FEATURE_COLS."
        )

    # Use last row values as next-day input features (common approach)
    last_row = df.iloc[-1][FEATURE_COLS]

    X_next = np.array(last_row, dtype=float).reshape(1, -1)

    # Fix NaN/inf for safety
    X_next = np.nan_to_num(X_next, nan=0.0, posinf=0.0, neginf=0.0)

    if X_next.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"Next-day feature count mismatch. Expected {len(FEATURE_COLS)} got {X_next.shape[1]}")

    if not np.isfinite(X_next).all():
        raise ValueError("Next-day features contain NaN/inf even after cleaning.")

    return X_next


def safe_predict(X: np.ndarray) -> float:
    """
    Predict and return JSON-safe float.
    """
    pred = model.predict(X)
    # Handles output like array([123.4]) or array([[...]])
    if hasattr(pred, "__len__"):
        return float(np.ravel(pred)[0])
    return float(pred)


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    # keeps your frontend as-is
    return render_template(TEMPLATE_NAME)


@app.post("/predict")
def predict():
    """
    Today prediction endpoint.
    Supports:
      - { "features": [ ... ] }
      - { "features_dict": { "Open":..., ... } }
    """
    try:
        payload = request.get_json(silent=True) or {}

        if "features_dict" in payload:
            X = build_features_from_dict(payload["features_dict"])
        elif "features" in payload:
            X = build_features_from_list(payload["features"])
        else:
            raise ValueError("Missing input. Send 'features' (array) OR 'features_dict' (object).")

        pred_value = safe_predict(X)

        return jsonify({"success": True, "prediction": pred_value})

    except Exception as e:
        print("❌ /predict error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. Please try again.",
            "debug_error": str(e)  # remove after debugging
        }), 500


@app.post("/predict-next-day")
def predict_next_day():
    """
    Next-day prediction endpoint (FIXED).
    Supports:
      - { "last_rows": [ {row1}, {row2}, ... ] }
      - OR fallback: { "features": [...] } / { "features_dict": {...} }
    """
    try:
        payload = request.get_json(silent=True) or {}

        if "last_rows" in payload and payload["last_rows"]:
            X_next = compute_next_day_X_from_history(payload["last_rows"])
        elif "features_dict" in payload:
            # If you don't send history, we just use given features
            X_next = build_features_from_dict(payload["features_dict"])
        elif "features" in payload:
            X_next = build_features_from_list(payload["features"])
        else:
            raise ValueError("Missing input. Send 'last_rows' OR 'features' OR 'features_dict'.")

        pred_value = safe_predict(X_next)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        return jsonify({
            "success": True,
            "date": tomorrow,
            "prediction": pred_value
        })

    except Exception as e:
        # This will show the real error in Railway logs
        print("❌ /predict-next-day error:", str(e))
        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. Please try again.",
            "debug_error": str(e)  # keep for now; remove later
        }), 500


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
