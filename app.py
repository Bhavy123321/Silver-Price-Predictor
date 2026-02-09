from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = "silver-predictor-secret"

# ---- always load latest CSS/JS on Railway ----
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

def predict(model, X):
    pred = int(model.predict(X)[0])
    proba_up = float(model.predict_proba(X)[0][1])
    direction = "UP" if pred == 1 else "DOWN"
    confidence = proba_up if pred == 1 else (1 - proba_up)
    return direction, round(confidence * 100, 2)

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            state = request.form["state"]
            horizon = request.form["horizon"]
            purity = request.form["purity"]

            silver_usd, usd_inr = fetch_market()
            if silver_usd is None:
                flash("Market data unavailable", "error")
                return redirect(url_for("index"))

            base_kg = usd_oz_to_inr_kg(silver_usd, usd_inr)
            premium = STATE_PREMIUM[state]
            purity_factor = SILVER_PURITY[purity]

            base_per_g = (base_kg / 1000.0) * purity_factor
            final_per_g = ((base_kg + premium) / 1000.0) * purity_factor

            X = [[silver_usd, usd_inr]]

            if horizon == "1h":
                direction, conf = predict(model_1h, X)
                horizon_label = "Next Hour"
            elif horizon == "1d":
                direction, conf = predict(model_1d, X)
                horizon_label = "Next Day (1d)"
            else:
                direction, conf = predict(model_1m, X)
                horizon_label = "Next Month"

            result = {
                "state": state,
                "horizon": horizon_label,
                "purity": purity,
                "direction": direction,
                "confidence": conf,
                "base_inr_kg": round(base_kg, 2),
                "premium_per_kg": premium,
                "base_per_g": round(base_per_g, 2),
                "final_per_g": round(final_per_g, 2),
                "prices": {
                    "p1": round(final_per_g * 1, 2),
                    "p10": round(final_per_g * 10, 2),
                    "p100": round(final_per_g * 100, 2),
                }
            }

        except Exception as e:
            print("ERROR:", e)
            flash("Something went wrong. Please try again.", "error")

    return render_template(
        "index.html",
        states=STATE_PREMIUM.keys(),
        result=result,
        social=SOCIAL
    )

@app.route("/about")
def about():
    return render_template("about.html", social=SOCIAL)

@app.route("/reviews")
def reviews():
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute("SELECT name, rating, message, created_at FROM reviews ORDER BY id DESC").fetchall()
    return render_template("reviews.html", reviews=rows, social=SOCIAL)

if __name__ == "__main__":
    app.run(debug=True)
