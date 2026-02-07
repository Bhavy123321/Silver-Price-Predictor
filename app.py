import os
import pickle
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# -----------------------------
# CONFIG (EDIT THESE 2)
# -----------------------------
MODEL_FILE = "model.pkl"          # <-- change to your actual .pkl file name
TEMPLATE_NAME = "index.html"      # <-- change if your template name is different

# MUST match training feature order exactly
FEATURE_COLS = [
    # Replace with your real features
    "Open", "High", "Low", "Close", "Volume",
    "MA_7", "MA_14", "RSI_14"
]

app = Flask(__name__)

# -----------------------------
# LOG REQUESTS (so you can see hits in Railway logs)
# -----------------------------
@app.before_request
def log_request():
    try:
        # Shows if Railway is actually reaching your app
        print(f"➡️ {request.method} {request.path}")
    except Exception:
        pass


# -----------------------------
# LOAD MODEL ONCE (fast + stable)
# -----------------------------
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print("❌ Model load failed:", str(e))
    print(traceback.format_exc())


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    # Railway routing/healthcheck test
    return "ok", 200


@app.get("/")
def home():
    """
    Keep this route super light.
    If template missing, still return a plain message (no hang).
    """
    try:
        return render_template(TEMPLATE_NAME)
    except Exception as e:
        print("⚠️ Template render failed:", str(e))
        return "App is running ✅ (template missing or error)", 200


# -----------------------------
# HELPERS
# -----------------------------
def safe_predict(X: np.ndarray) -> float:
    if model is None:
        raise RuntimeError("Model is not loaded. Check model file name/path.")
    pred = model.predict(X)
    return float(np.ravel(pred)[0])


def build_X_from_payload(payload: dict) -> np.ndarray:
    """
    Supports:
      - payload["features"] = [....]
      - payload["features_dict"] = { "Open":..., ... }
    """
    if "features_dict" in payload and payload["features_dict"] is not None:
        fd = payload["features_dict"]
        if not isinstance(fd, dict):
            raise ValueError("features_dict must be an object")

        row = []
        for col in FEATURE_COLS:
            if col not in fd:
                raise ValueError(f"Missing feature '{col}' in features_dict")
            row.append(float(fd[col]))

        X = np.array(row, dtype=float).reshape(1, -1)

    elif "features" in payload and payload["features"] is not None:
        fl = payload["features"]
        if not isinstance(fl, list):
            raise ValueError("features must be an array")

        X = np.array(fl, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

    else:
        raise ValueError("Send 'features' (array) OR 'features_dict' (object).")

    if X.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"Feature count mismatch. Expected {len(FEATURE_COLS)} got {X.shape[1]}")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN/inf.")

    return X


def compute_next_day_X(last_rows: list) -> np.ndarray:
    """
    Builds next-day features safely from history rows.
    last_rows: list of dict rows (from frontend)
    """
    if not last_rows or not isinstance(last_rows, list):
        raise ValueError("last_rows is missing or invalid")

    df = pd.DataFrame(last_rows)
    if df.empty:
        raise ValueError("History dataframe is empty")

    # Ensure numeric base columns if present
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute indicators ONLY if your feature list requires them
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
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            df["RSI_14"] = rsi.fillna(50)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for next-day: {missing}")

    last = df.iloc[-1][FEATURE_COLS]
    X_next = np.array(last, dtype=float).reshape(1, -1)
    X_next = np.nan_to_num(X_next, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(X_next).all():
        raise ValueError("Next-day features still contain NaN/inf after cleaning.")

    return X_next


# -----------------------------
# API: Predict Today
# -----------------------------
@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or {}
        X = build_X_from_payload(payload)
        pred_value = safe_predict(X)
        return jsonify({"success": True, "prediction": pred_value})
    except Exception as e:
        print("❌ /predict error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. Please try again.",
            "debug_error": str(e)
        }), 500


# -----------------------------
# API: Predict Next Day
# -----------------------------
@app.post("/predict-next-day")
def predict_next_day():
    try:
        payload = request.get_json(silent=True) or {}

        if "last_rows" in payload and payload["last_rows"]:
            X_next = compute_next_day_X(payload["last_rows"])
        else:
            # Fallback: if frontend doesn't send history, use direct features
            X_next = build_X_from_payload(payload)

        pred_value = safe_predict(X_next)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        return jsonify({"success": True, "date": tomorrow, "prediction": pred_value})

    except Exception as e:
        print("❌ /predict-next-day error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. Please try again.",
            "debug_error": str(e)
        }), 500


# -----------------------------
# LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
