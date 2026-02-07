import os
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from utils import load_model_and_meta, build_next_day_features_safe

app = Flask(__name__)

# Loads model + metadata (feature columns) safely
MODEL, META = load_model_and_meta()

# -----------------------
# Pages
# -----------------------
@app.get("/")
def home():
    return render_template("index.html")


# -----------------------
# APIs
# -----------------------
@app.post("/predict")
def predict():
    """
    Predict current/day price movement/value based on user inputs.
    Your frontend likely calls this.
    """
    try:
        payload = request.get_json(silent=True) or {}

        # Expecting frontend sends features as dict OR list.
        # We'll support both.
        features_dict = payload.get("features_dict")
        features_list = payload.get("features")

        if features_dict is None and features_list is None:
            raise ValueError("Missing input: send either 'features_dict' (object) or 'features' (array).")

        X = None
        if features_dict is not None:
            # Ensure correct column order using META["feature_names"]
            if not META.get("feature_names"):
                raise RuntimeError("feature_names missing in meta. Add meta.json or set FEATURE_NAMES in utils.py.")

            row = []
            for col in META["feature_names"]:
                if col not in features_dict:
                    raise ValueError(f"Missing feature '{col}' in request.")
                row.append(float(features_dict[col]))
            X = np.array(row, dtype=float).reshape(1, -1)
        else:
            X = np.array(features_list, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        # Validate
        if X.shape[1] != len(META["feature_names"]):
            raise ValueError(f"Feature count mismatch. Expected {len(META['feature_names'])}, got {X.shape[1]}")

        if not np.isfinite(X).all():
            raise ValueError("Inputs contain NaN/inf. Please fill all fields correctly.")

        y_pred = MODEL.predict(X)
        pred_value = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)

        return jsonify({"success": True, "prediction": pred_value})

    except Exception as e:
        print("❌ /predict failed:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction. please try again",
            "debug_error": str(e)  # remove later
        }), 500


@app.post("/predict-next-day")
def predict_next_day():
    """
    Predict next day using last known data and safe feature-building.
    This is where your error is happening.
    """
    try:
        payload = request.get_json(silent=True) or {}

        # Expect either:
        # 1) last_rows: list of dicts (history), OR
        # 2) last_row_features_dict/features: if you already have the last row inputs
        last_rows = payload.get("last_rows")

        if last_rows is None:
            # If frontend isn't sending history, we can still attempt using direct features
            features_dict = payload.get("features_dict")
            features_list = payload.get("features")
            if features_dict is None and features_list is None:
                raise ValueError("Missing input. Send 'last_rows' OR 'features_dict'/'features'.")
            # Build next-day features from single row (no rolling)
            X_next = build_next_day_features_safe(
                last_df=None,
                feature_names=META["feature_names"],
                features_dict=features_dict,
                features_list=features_list
            )
        else:
            # Build from dataframe history
            df = pd.DataFrame(last_rows)
            if df.empty:
                raise ValueError("last_rows is empty; cannot calculate next-day features.")

            X_next = build_next_day_features_safe(
                last_df=df,
                feature_names=META["feature_names"],
                features_dict=None,
                features_list=None
            )

        # Validate model + X_next
        if MODEL is None:
            raise RuntimeError("Model not loaded (MODEL is None).")

        if X_next.shape[1] != len(META["feature_names"]):
            raise ValueError(f"Next-day feature mismatch. Expected {len(META['feature_names'])}, got {X_next.shape[1]}")

        if not np.isfinite(X_next).all():
            raise ValueError("Next-day features contain NaN/inf (rolling/lag issue).")

        y_pred = MODEL.predict(X_next)
        pred_value = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)

        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        return jsonify({
            "success": True,
            "date": tomorrow,
            "prediction": pred_value
        })

    except Exception as e:
        print("❌ /predict-next-day failed:", str(e))
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "message": "Something went wrong while calculating prediction.please try again",
            "debug_error": str(e)  # keep for now to see exact issue
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
