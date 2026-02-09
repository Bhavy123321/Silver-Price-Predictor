from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import yfinance as yf
import sqlite3
import os
from datetime import datetime
import time
import random

app = Flask(__name__)
app.secret_key = "silver-predictor-secret"

# Always load newest CSS/JS
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
# LOAD MODELS (safe load)
# -------------------------
def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print("MODEL LOAD ERROR:", path, e)
        return None

model_1h = safe_load_model("models/model_next_hour.joblib")
model_1d = safe_load_model("models/model_next_day.joblib")
model_1m = safe_load_model("models/model_next_month.joblib")

# -------------------------
# DATABASE (REVIEWS)
# -------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "reviews.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS reviews(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
init_db()

def add_review(name, rating, message):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO reviews(name, rating, message, created_at) VALUES(?,?,?,?)",
            (name, rating, message, datetime.now().strftime("%Y-%m-%d %H:%M"))
        )

def get_reviews(limit=80):
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT name, rating, message, created_at FROM reviews ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return rows

# -------------------------
# HELPERS
# -------------------------
def usd_oz_to_inr_kg(price_usd_oz, usd_inr):
    return price_usd_oz * usd_inr * (1000 / 31.1035)

def fetch_market_safe():
    """
    1) Try Yahoo Finance
    2) If it fails, use fallback constants so app NEVER breaks.
    """
    try:
        silver = yf.download("SI=F", period="5d", interval="1d", progress=False, threads=False)
        usd = yf.download("USDINR=X", period="5d", interval="1d", progress=False, threads=False)

        if silver is None or usd is None or silver.empty or usd.empty:
            raise ValueError("Empty market data")

        silver_close = float(silver["Close"].dropna().iloc[-1])
        usd_close = float(usd["Close"].dropna().iloc[-1])

        return silver_close, usd_close, True  # live=True
    except Exception as e:
        print("MARKET FETCH ERROR:", e)
        fallback_silver_usd_oz = 24.5
        fallback_usd_inr = 83.0
        return fallback_silver_usd_oz, fallback_usd_inr, False

def predict_safe(model, X):
    """
    If model is missing, still return a fallback prediction.
    """
    try:
        if model is None:
            return "UP", 70.0, False
        pred = int(model.predict(X)[0])
        proba_up = float(model.predict_proba(X)[0][1])
        direction = "UP" if pred == 1 else "DOWN"
        conf = proba_up if pred == 1 else (1 - proba_up)
        return direction, round(conf * 100, 2), True
    except Exception as e:
        print("PREDICT ERROR:", e)
        return "UP", 70.0, False

def make_fallback_series(days=30, base=250000.0):
    """Creates a smooth demo line (₹/kg) so chart never stays blank."""
    vals = []
    cur = base
    for _ in range(days):
        cur += random.uniform(-2500, 2500)  # gentle move
        vals.append(round(cur, 2))
    return vals

# -------------------------
# API: Trend for chart (₹/kg)
# -------------------------
@app.route("/api/trend")
def api_trend():
    try:
        # 30-day daily series
        silver = yf.download("SI=F", period="1mo", interval="1d", progress=False, threads=False)
        usd = yf.download("USDINR=X", period="1mo", interval="1d", progress=False, threads=False)

        if silver is None or usd is None or silver.empty or usd.empty:
            raise ValueError("Empty market data")

        silver = silver[["Close"]].dropna()
        usd = usd[["Close"]].dropna()

        # align by index
        joined = silver.join(usd, how="inner", lsuffix="_silver", rsuffix="_usd")
        joined = joined.dropna()

        labels = [idx.strftime("%d %b") for idx in joined.index]
        values = []
        for _, row in joined.iterrows():
            inr_kg = usd_oz_to_inr_kg(float(row["Close_silver"]), float(row["Close_usd"]))
            values.append(round(inr_kg, 2))

        if len(values) < 5:
            raise ValueError("Not enough points")

        return jsonify({"ok": True, "labels": labels[-30:], "values": values[-30:], "live": True})

    except Exception as e:
        print("API TREND ERROR:", e)
        # fallback: base from current safe market
        silver_usd, usd_inr, _ = fetch_market_safe()
        base = usd_oz_to_inr_kg(silver_usd, usd_inr)
        labels = [(datetime.now().date()).strftime("%d %b")]
        # create labels for last 30 days
        labels = []
        for i in range(29, -1, -1):
            d = datetime.now().date()
            labels.append((d.replace(day=d.day) - __import__("datetime").timedelta(days=i)).strftime("%d %b"))
        values = make_fallback_series(30, base=base)
        return jsonify({"ok": True, "labels": labels, "values": values, "live": False})

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        state = request.form.get("state", "").strip()
        horizon = request.form.get("horizon", "").strip()
        purity = request.form.get("purity", "").strip()

        if not state or not horizon or not purity:
            flash("Please select State/UT, Horizon and Purity.", "error")
            return redirect(url_for("index"))

        try:
            silver_usd, usd_inr, is_live = fetch_market_safe()

            base_kg = usd_oz_to_inr_kg(silver_usd, usd_inr)
            premium = STATE_PREMIUM.get(state, 0)
            purity_factor = SILVER_PURITY.get(purity, 1.0)

            base_per_g = (base_kg / 1000.0) * purity_factor
            final_per_g = ((base_kg + premium) / 1000.0) * purity_factor

            X = [[silver_usd, usd_inr]]

            if horizon == "1h":
                direction, conf, model_ok = predict_safe(model_1h, X)
                horizon_label = "Next Hour"
            elif horizon == "1d":
                direction, conf, model_ok = predict_safe(model_1d, X)
                horizon_label = "Next Day (1d)"
            else:
                direction, conf, model_ok = predict_safe(model_1m, X)
                horizon_label = "Next Month"

            # Not error — friendly info
            if not is_live:
                flash("Live market feed blocked. Showing estimate using fallback values.", "success")
            if not model_ok:
                flash("Model load issue. Showing a safe demo prediction output.", "success")

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
                    "p100": round(final_per_g * 100, 2)
                }
            }

        except Exception as e:
            print("INDEX ERROR:", e)
            flash("Something went wrong. Please try again.", "error")
            return redirect(url_for("index"))

    # premium chart data (top 8)
    top_prem = sorted(STATE_PREMIUM.items(), key=lambda x: x[1], reverse=True)[:8]
    prem_labels = [k for k, _ in top_prem]
    prem_values = [v for _, v in top_prem]

    return render_template(
        "index.html",
        states=STATE_PREMIUM.keys(),
        result=result,
        social=SOCIAL,
        prem_labels=prem_labels,
        prem_values=prem_values
    )

@app.route("/about")
def about():
    return render_template("about.html", social=SOCIAL)

@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    if request.method == "POST":
        try:
            name = (request.form.get("name") or "").strip()
            rating = int(request.form.get("rating") or 5)
            message = (request.form.get("message") or "").strip()

            if not name or not message:
                flash("Please enter your name and message.", "error")
                return redirect(url_for("reviews"))

            if rating < 1 or rating > 5:
                flash("Rating must be between 1 and 5.", "error")
                return redirect(url_for("reviews"))

            add_review(name, rating, message)
            flash("Review submitted successfully!", "success")
            return redirect(url_for("reviews"))

        except Exception as e:
            print("REVIEW ERROR:", e)
            flash("Could not submit review. Please try again.", "error")
            return redirect(url_for("reviews"))

    rows = get_reviews()
    return render_template("reviews.html", reviews=rows, social=SOCIAL)

if __name__ == "__main__":
    app.run(debug=True)
