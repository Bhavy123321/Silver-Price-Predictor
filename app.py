from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import yfinance as yf
import sqlite3
import os
import time
from datetime import datetime
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
app.secret_key = "silver-predictor-secret"

# -------------------------
# CACHE BUST (forces latest CSS/JS)
# -------------------------
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.context_processor
def inject_cache_bust():
    return {"cache_bust": int(time.time())}

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -------------------------
# SOCIAL LINKS
# -------------------------
SOCIAL = {
    "github": "https://github.com/YOUR_GITHUB",
    "linkedin": "https://linkedin.com/in/YOUR_LINKEDIN"
}

# -------------------------
# STATE PREMIUMS (₹/kg)
# -------------------------
STATE_PREMIUM = {
    "Andaman and Nicobar Islands": 650,
    "Andhra Pradesh": 700,
    "Arunachal Pradesh": 600,
    "Assam": 650,
    "Bihar": 750,
    "Chandigarh": 850,
    "Chhattisgarh": 700,
    "Delhi": 900,
    "Goa": 650,
    "Gujarat": 800,
    "Haryana": 850,
    "Himachal Pradesh": 700,
    "Jammu and Kashmir": 750,
    "Jharkhand": 700,
    "Karnataka": 650,
    "Kerala": 600,
    "Madhya Pradesh": 720,
    "Maharashtra": 1000,
    "Odisha": 700,
    "Punjab": 850,
    "Rajasthan": 750,
    "Tamil Nadu": 600,
    "Telangana": 650,
    "Uttar Pradesh": 850,
    "Uttarakhand": 800,
    "West Bengal": 650
}

# -------------------------
# SILVER PURITY FACTOR
# -------------------------
SILVER_PURITY = {
    "999": 1.0,
    "925": 0.925,
    "900": 0.9,
    "800": 0.8
}

# -------------------------
# LOAD MODELS
# -------------------------
model_1h = joblib.load("models/model_next_hour.joblib")
model_1d = joblib.load("models/model_next_day.joblib")
model_1m = joblib.load("models/model_next_month.joblib")

# -------------------------
# DATABASE (REVIEWS)
# -------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "reviews.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS reviews(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            rating INTEGER,
            message TEXT,
            created_at TEXT
        )
        """)
init_db()

# -------------------------
# HELPERS
# -------------------------
def usd_oz_to_inr_kg(price_usd_oz, usd_inr):
    return price_usd_oz * usd_inr * (1000 / 31.1035)

def fetch_market():
    silver = yf.download("SI=F", period="5d", interval="1d", progress=False)
    usd = yf.download("USDINR=X", period="5d", interval="1d", progress=False)

    if silver.empty or usd.empty:
        return None, None

    return float(silver["Close"].iloc[-1]), float(usd["Close"].iloc[-1])

def predict_direction(model, X):
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return ("UP" if pred == 1 else "DOWN", round(proba * 100, 2))

# -------------------------
# PAGES
# -------------------------
@app.route("/", methods=["GET"])
def index():
    # JS-based UI; no POST form needed now
    return render_template("index.html", states=sorted(STATE_PREMIUM.keys()), social=SOCIAL)

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", social=SOCIAL)

@app.route("/reviews", methods=["GET"])
def reviews():
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT name, rating, message, created_at FROM reviews ORDER BY id DESC").fetchall()
    return render_template("reviews.html", reviews=rows, social=SOCIAL)

# -------------------------
# API: Trend for Chart
# -------------------------
@app.route("/api/trend", methods=["GET"])
def api_trend():
    """
    Returns last ~48 hours (hourly) silver close values from yfinance.
    If yfinance fails, returns a fallback smooth series.
    """
    try:
        df = yf.download("SI=F", period="2d", interval="60m", progress=False)
        if df is None or df.empty:
            raise ValueError("No trend data")

        closes = df["Close"].dropna()
        closes = closes.tail(48)

        labels = [ts.strftime("%H:%M") for ts in closes.index]
        values = [round(float(v), 2) for v in closes.values]

        return jsonify({"ok": True, "labels": labels, "values": values})
    except Exception:
        # fallback demo series
        labels = []
        values = []
        v = 95.0
        for i in range(48):
            labels.append(f"{i:02d}:00")
            v += (0.6 if i % 7 else -3.2)
            v = max(70, min(120, v))
            values.append(round(v, 2))
        return jsonify({"ok": True, "labels": labels, "values": values})

# -------------------------
# API: Prediction (JSON) used by app.js
# -------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True) or {}
        state = (data.get("state") or "").strip()
        purity = (data.get("purity") or "").strip()
        horizon = (data.get("horizon") or "").strip()  # "1h" | "1d" | "1m"

        if not state or not purity or not horizon:
            return jsonify({"ok": False, "error": "Please select State/UT, Purity and Horizon."}), 400

        silver_usd, usd_inr = fetch_market()
        if silver_usd is None:
            return jsonify({"ok": False, "error": "Market data unavailable right now. Try again."}), 503

        if state not in STATE_PREMIUM:
            return jsonify({"ok": False, "error": "Invalid state selected."}), 400
        if purity not in SILVER_PURITY:
            return jsonify({"ok": False, "error": "Invalid purity selected."}), 400

        base_kg = usd_oz_to_inr_kg(silver_usd, usd_inr)
        premium = STATE_PREMIUM[state]
        purity_factor = SILVER_PURITY[purity]

        final_per_g = ((base_kg + premium) / 1000) * purity_factor

        X = [[silver_usd, usd_inr]]

        if horizon == "1h":
            direction, conf = predict_direction(model_1h, X)
            horizon_label = "Next Hour"
        elif horizon == "1d":
            direction, conf = predict_direction(model_1d, X)
            horizon_label = "Next Day"
        else:
            direction, conf = predict_direction(model_1m, X)
            horizon_label = "Next Month"

        return jsonify({
            "ok": True,
            "direction": direction,
            "confidence": conf,  # percent (0-100)
            "meta": {"state": state, "purity": purity, "horizon": horizon_label},
            "prices": {
                "1g": round(final_per_g, 2),
                "10g": round(final_per_g * 10, 2),
                "100g": round(final_per_g * 100, 2),
            }
        })

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"ok": False, "error": "Something went wrong while calculating prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True)
