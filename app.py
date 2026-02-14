from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import yfinance as yf
import sqlite3
import os
from datetime import datetime, timedelta
import json
import math
import random
import time

app = Flask(__name__)
app.secret_key = "silver-predictor-secret"

# Ensure templates reload while developing
app.config["TEMPLATES_AUTO_RELOAD"] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "reviews.db")
CACHE_PATH = os.path.join(BASE_DIR, "market_cache.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")


# -------------------------
# Add cache-busting for static files (CSS/JS)
# -------------------------
@app.context_processor
def inject_cache_bust():
    # Changes each server boot; enough to force refresh on deploy
    return {"cache_bust": int(time.time())}


# -------------------------
# SOCIAL LINKS
# -------------------------
SOCIAL = {
    "github": "https://github.com/Bhavy123321",
    "linkedin": "https://www.linkedin.com/in/bhavy-soni-6123a32b0/"
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
# DATABASE (REVIEWS)
# -------------------------
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
# MODELS (absolute path fix)
# -------------------------
def safe_load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    try:
        return joblib.load(path)
    except Exception as e:
        print("MODEL LOAD ERROR:", path, e)
        return None

model_1h = safe_load_model("model_next_hour.joblib")
model_1d = safe_load_model("model_next_day.joblib")
model_1m = safe_load_model("model_next_month.joblib")


# -------------------------
# MARKET CACHE
# -------------------------
def read_market_cache():
    try:
        if not os.path.exists(CACHE_PATH):
            return None
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("CACHE READ ERROR:", e)
        return None

def write_market_cache(silver_usd_oz, usd_inr):
    try:
        payload = {
            "silver_usd_oz": float(silver_usd_oz),
            "usd_inr": float(usd_inr),
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        print("CACHE WRITE ERROR:", e)

def parse_saved_at(saved_at):
    try:
        return datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
    except:
        return None


# -------------------------
# HELPERS
# -------------------------
def usd_oz_to_inr_kg(price_usd_oz, usd_inr):
    # 1 troy ounce = 31.1035g, 1kg=1000g
    return price_usd_oz * usd_inr * (1000 / 31.1035)

def fetch_market_best():
    """
    1) Try LIVE Yahoo (yfinance)
    2) If failed -> use LAST SUCCESSFUL cached value (accurate enough)
    3) If no cache -> use realistic fallback (last resort)
    Returns: silver_usd_oz, usd_inr, source ("live" | "cached" | "fallback")
    """
    # 1) live
    try:
        silver = yf.download("SI=F", period="5d", interval="1d", progress=False, threads=False)
        usd = yf.download("USDINR=X", period="5d", interval="1d", progress=False, threads=False)

        if silver is None or usd is None or silver.empty or usd.empty:
            raise ValueError("Empty market data")

        silver_close = float(silver["Close"].dropna().iloc[-1])
        usd_close = float(usd["Close"].dropna().iloc[-1])

        write_market_cache(silver_close, usd_close)
        return silver_close, usd_close, "live"

    except Exception as e:
        print("LIVE MARKET ERROR:", e)

    # 2) cached
    cache = read_market_cache()
    if cache:
        saved_at = parse_saved_at(cache.get("saved_at", ""))
        if saved_at:
            # allow cached up to 14 days
            age = datetime.now(saved_at.tzinfo) - saved_at
            if age <= timedelta(days=14):
                try:
                    return float(cache["silver_usd_oz"]), float(cache["usd_inr"]), "cached"
                except:
                    pass

    # 3) fallback (last resort) — REALISTIC
    # Your expectation: ~₹265,000/kg.
    # If silver is ~₹265/g, then INR/oz ≈ 265 * 31.1035 ≈ 8242 INR/oz.
    # USD/oz ≈ 8242 / 83 ≈ 99.3 USD/oz
    # This keeps fallback close to real-world INR/kg.
    return 99.3, 83.0, "fallback"


def predict_safe(model, X):
    """
    If model missing, show demo direction/confidence instead of crashing.
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


def calc_daily_volatility_inrkg():
    """
    Daily volatility from 30-day INR/kg series.
    If Yahoo fails -> safe default.
    """
    try:
        silver = yf.download("SI=F", period="1mo", interval="1d", progress=False, threads=False)
        usd = yf.download("USDINR=X", period="1mo", interval="1d", progress=False, threads=False)

        if silver is None or usd is None or silver.empty or usd.empty:
            raise ValueError("Empty data")

        silver = silver[["Close"]].dropna()
        usd = usd[["Close"]].dropna()
        joined = silver.join(usd, how="inner", lsuffix="_silver", rsuffix="_usd").dropna()

        values = []
        for _, row in joined.iterrows():
            inrkg = usd_oz_to_inr_kg(float(row["Close_silver"]), float(row["Close_usd"]))
            values.append(inrkg)

        if len(values) < 8:
            raise ValueError("Not enough points")

        rets = []
        for i in range(1, len(values)):
            rets.append((values[i] - values[i-1]) / values[i-1])

        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)

    except Exception as e:
        print("VOL ERROR:", e)
        return 0.012  # ~1.2% daily fallback


def estimate_move_pct(horizon_key, confidence_pct, direction):
    """
    Converts confidence + volatility into expected move % (capped).
    """
    daily_vol = calc_daily_volatility_inrkg()
    conf = max(0.0, min(100.0, float(confidence_pct))) / 100.0

    if horizon_key == "1h":
        vol = daily_vol / math.sqrt(24)
        cap = 0.05
    elif horizon_key == "1d":
        vol = daily_vol
        cap = 0.12
    else:
        vol = daily_vol * math.sqrt(30)
        cap = 0.25

    raw = conf * vol * 2.2
    move = max(-cap, min(cap, raw))
    sign = 1 if direction == "UP" else -1
    return sign * move


# -------------------------
# API: Trend Chart
# -------------------------
@app.route("/api/trend")
def api_trend():
    try:
        silver = yf.download("SI=F", period="1mo", interval="1d", progress=False, threads=False)
        usd = yf.download("USDINR=X", period="1mo", interval="1d", progress=False, threads=False)
        if silver is None or usd is None or silver.empty or usd.empty:
            raise ValueError("Empty market data")

        silver = silver[["Close"]].dropna()
        usd = usd[["Close"]].dropna()
        joined = silver.join(usd, how="inner", lsuffix="_silver", rsuffix="_usd").dropna()

        labels = [idx.strftime("%d %b") for idx in joined.index]
        values = []
        for _, row in joined.iterrows():
            inr_kg = usd_oz_to_inr_kg(float(row["Close_silver"]), float(row["Close_usd"]))
            values.append(round(inr_kg, 2))

        return jsonify({"ok": True, "labels": labels[-30:], "values": values[-30:], "source": "live"})

    except Exception as e:
        print("API TREND ERROR:", e)
        silver_usd, usd_inr, src = fetch_market_best()
        base = usd_oz_to_inr_kg(silver_usd, usd_inr)
        labels = [(datetime.now().date() - timedelta(days=i)).strftime("%d %b") for i in range(29, -1, -1)]
        vals = []
        cur = base
        for _ in range(30):
            cur += random.uniform(-2500, 2500)
            vals.append(round(cur, 2))
        return jsonify({"ok": True, "labels": labels, "values": vals, "source": src})


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

        silver_usd, usd_inr, market_source = fetch_market_best()

        base_kg = usd_oz_to_inr_kg(silver_usd, usd_inr)
        premium = STATE_PREMIUM.get(state, 0)
        purity_factor = SILVER_PURITY.get(purity, 1.0)

        # Current reference (State premium + purity)
        current_per_g = ((base_kg + premium) / 1000.0) * purity_factor
        current_per_kg = (base_kg + premium) * purity_factor

        X = [[silver_usd, usd_inr]]
        if horizon == "1h":
            direction, conf, model_ok = predict_safe(model_1h, X)
            horizon_label = "Next Hour"
            horizon_key = "1h"
        elif horizon == "1d":
            direction, conf, model_ok = predict_safe(model_1d, X)
            horizon_label = "Next Day (1d)"
            horizon_key = "1d"
        else:
            direction, conf, model_ok = predict_safe(model_1m, X)
            horizon_label = "Next Month"
            horizon_key = "1m"

        # Predicted price estimate
        move_pct = estimate_move_pct(horizon_key, conf, direction)
        predicted_per_g = current_per_g * (1.0 + move_pct)
        predicted_per_kg = current_per_kg * (1.0 + move_pct)

        result = {
            "state": state,
            "horizon": horizon_label,
            "purity": purity,
            "direction": direction,
            "confidence": conf,
            "market_source": market_source,  # live/cached/fallback

            "current_per_g": round(current_per_g, 2),
            "current_per_kg": round(current_per_kg, 2),
            "predicted_per_g": round(predicted_per_g, 2),
            "predicted_per_kg": round(predicted_per_kg, 2),

            "prices": {
                "p1": round(predicted_per_g * 1, 2),
                "p10": round(predicted_per_g * 10, 2),
                "p100": round(predicted_per_g * 100, 2),
            },
            "move_pct": round(move_pct * 100, 2),
            "model_ok": model_ok
        }

    # chart data: top 8 premiums
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
    
@app.context_processor
def inject_globals():
    return dict(SOCIAL=SOCIAL)
    
@app.route("/about")
def about():
    return render_template("about.html", title="About")
    
    

@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    if request.method == "POST":
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

    rows = get_reviews()
    return render_template("reviews.html", reviews=rows, social=SOCIAL)

if __name__ == "__main__":
    app.run(debug=True)


