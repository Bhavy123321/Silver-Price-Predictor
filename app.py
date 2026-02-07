from flask import Flask, render_template, request, jsonify
import os
import math
from datetime import datetime, timedelta
import random

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# Helpers (simple demo logic)
# Replace this with your real ML model prediction
# -----------------------------
def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def estimate_prices(base_price_per_kg, direction_up: bool):
    """
    Estimate prices for 1g/10g/100g from a base per-kg price.
    This is just a clean UX helper. Replace if you have exact logic.
    """
    # If base_price_per_kg is INR per kg:
    price_1g = base_price_per_kg / 1000.0
    price_10g = price_1g * 10
    price_100g = price_1g * 100

    # Add tiny adjustment based on direction for UI feel
    tweak = 0.006 if direction_up else -0.006
    price_1g *= (1 + tweak)
    price_10g *= (1 + tweak)
    price_100g *= (1 + tweak)

    return {
        "1g": round(price_1g, 2),
        "10g": round(price_10g, 2),
        "100g": round(price_100g, 2),
    }

def make_trend_series(hours=48, start_value=98.0):
    """
    Create a smooth-ish trend line for the chart (demo).
    Replace with your real data source if you have it.
    """
    now = datetime.utcnow()
    labels = []
    values = []
    v = start_value

    for i in range(hours):
        t = now - timedelta(hours=(hours - 1 - i))
        labels.append(t.strftime("%H:%M"))
        # gentle random walk
        v += random.uniform(-1.4, 1.2)
        v = max(70, min(120, v))
        values.append(round(v, 2))

    return labels, values

# -----------------------------
# Pages
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/reviews", methods=["GET"])
def reviews():
    return render_template("reviews.html")

# -----------------------------
# APIs
# -----------------------------
@app.route("/api/trend", methods=["GET"])
def api_trend():
    labels, values = make_trend_series(hours=48, start_value=105.0)
    return jsonify({"labels": labels, "values": values})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "state": "Gujarat",
      "purity": "999",
      "horizon": "Next Hour"
    }
    """
    try:
        data = request.get_json(force=True) or {}

        state = (data.get("state") or "").strip()
        purity = (data.get("purity") or "").strip()
        horizon = (data.get("horizon") or "").strip()

        if not state or not purity or not horizon:
            return jsonify({
                "ok": False,
                "error": "Please select State/UT, Purity, and Horizon."
            }), 400

        # -----------------------------
        # Replace this block with your real ML model logic
        # -----------------------------
        # Demo: create a deterministic-ish decision from inputs
        key = f"{state}|{purity}|{horizon}".lower()
        score = sum(ord(c) for c in key) % 100
        direction_up = score >= 50
        confidence = 0.60 + (abs(score - 50) / 100.0) * 0.35  # 0.60 -> 0.95
        confidence = round(min(0.95, max(0.55, confidence)), 2)

        # Demo: base price INR/kg-ish
        base_price_per_kg = 85000 + (score * 120)  # 85k..97k
        prices = estimate_prices(base_price_per_kg, direction_up)

        return jsonify({
            "ok": True,
            "direction": "UP" if direction_up else "DOWN",
            "confidence": confidence,
            "prices": prices,
            "meta": {
                "state": state,
                "purity": purity,
                "horizon": horizon
            }
        })
    except Exception:
        # IMPORTANT: don’t show error banner on initial load
        return jsonify({
            "ok": False,
            "error": "Something went wrong while calculating prediction. Please try again."
        }), 500


if __name__ == "__main__":
    # Railway uses PORT env variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
