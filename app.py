from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import yfinance as yf
import pandas as pd
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "silver-predictor-secret"

# -------------------------
# YOUR SOCIAL LINKS (edit)
# -------------------------
SOCIAL = {
    "github": "https://github.com/Bhavy123321",
    "linkedin": "https://www.linkedin.com/in/bhavy-soni-6123a32b0/"
}

# -------------------------
# ALL INDIA: States + UTs
# Premium here is a simple demo premium (₹ per kg)
# You can tune these based on real market research later.
# -------------------------
STATE_PREMIUM = {
    # States
    "Andhra Pradesh": 700,
    "Arunachal Pradesh": 600,
    "Assam": 650,
    "Bihar": 750,
    "Chhattisgarh": 700,
    "Goa": 650,
    "Gujarat": 800,
    "Haryana": 850,
    "Himachal Pradesh": 700,
    "Jharkhand": 700,
    "Karnataka": 650,
    "Kerala": 600,
    "Madhya Pradesh": 720,
    "Maharashtra": 1000,
    "Manipur": 600,
    "Meghalaya": 600,
    "Mizoram": 600,
    "Nagaland": 600,
    "Odisha": 700,
    "Punjab": 850,
    "Rajasthan": 750,
    "Sikkim": 600,
    "Tamil Nadu": 600,
    "Telangana": 650,
    "Tripura": 600,
    "Uttar Pradesh": 850,
    "Uttarakhand": 800,
    "West Bengal": 650,

    # Union Territories
    "Andaman and Nicobar Islands": 650,
    "Chandigarh": 850,
    "Dadra and Nagar Haveli and Daman and Diu": 700,
    "Delhi": 900,
    "Jammu and Kashmir": 750,
    "Ladakh": 720,
    "Lakshadweep": 650,
    "Puducherry": 650,
}

# -------------------------
# Silver Purity Factors
# -------------------------
SILVER_PURITY = {
    "999": 1.00,   # Fine Silver
    "925": 0.925,  # Sterling
    "900": 0.90,
    "800": 0.80
}

# -------------------------
# Load ML Models
# -------------------------
model_1h = joblib.load("models/model_next_hour.joblib")
model_1d = joblib.load("models/model_next_day.joblib")
model_1m = joblib.load("models/model_next_month.joblib")

# -------------------------
# SQLite Reviews DB
# -------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "reviews.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

init_db()

def add_review(name: str, rating: int, message: str):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO reviews (name, rating, message, created_at) VALUES (?, ?, ?, ?)",
        (name, rating, message, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    con.commit()
    con.close()

def get_reviews(limit=60):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT name, rating, message, created_at FROM reviews ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return rows

# -------------------------
# Helpers
# -------------------------
def usd_oz_to_inr_kg(price_usd_per_oz, usd_inr):
    # 1 troy ounce = 31.1034768 grams
    oz_per_kg = 1000.0 / 31.1034768
    return float(price_usd_per_oz) * float(usd_inr) * oz_per_kg

def predict_with_model(model, X_row):
    pred = int(model.predict(X_row)[0])
    proba_up = float(model.predict_proba(X_row)[0][1])
    direction = "UP" if pred == 1 else "DOWN"
    confidence = proba_up if pred == 1 else (1 - proba_up)
    return direction, round(confidence * 100, 2), round(proba_up * 100, 2)

def safe_close_series(df):
    # yfinance can return MultiIndex columns sometimes
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def fetch_hourly_series(days="5d"):
    try:
        silver = yf.download("SI=F", period=days, interval="1h", auto_adjust=True, progress=False)
        series = safe_close_series(silver)
        return series
    except Exception:
        return None

def fetch_hourly_features():
    silver = yf.download("SI=F", period="5d", interval="1h", auto_adjust=True, progress=False)
    usdinr = yf.download("USDINR=X", period="5d", interval="1h", auto_adjust=True, progress=False)

    silver_close = safe_close_series(silver)
    usd_inr = safe_close_series(usdinr)

    if silver_close is None or usd_inr is None:
        return None, None

    df = pd.concat([silver_close.rename("silver_close"), usd_inr.rename("usd_inr")], axis=1).dropna()
    df["lag_1"] = df["silver_close"].shift(1)
    df["lag_2"] = df["silver_close"].shift(2)
    df["lag_3"] = df["silver_close"].shift(3)
    df["hour"] = df.index.hour
    df = df.dropna()

    if df.empty:
        return None, None

    X_last = df[["silver_close", "usd_inr", "lag_1", "lag_2", "lag_3", "hour"]].iloc[[-1]]
    latest = df.iloc[-1]
    return X_last, latest

def fetch_daily_features_for_month():
    silver = yf.download("SI=F", period="2y", interval="1d", auto_adjust=True, progress=False)
    usdinr = yf.download("USDINR=X", period="2y", interval="1d", auto_adjust=True, progress=False)

    silver_close = safe_close_series(silver)
    usd_inr = safe_close_series(usdinr)

    if silver_close is None or usd_inr is None:
        return None, None

    df = pd.concat([silver_close.rename("silver_close"), usd_inr.rename("usd_inr")], axis=1).dropna()

    df["lag_1"]  = df["silver_close"].shift(1)
    df["lag_5"]  = df["silver_close"].shift(5)
    df["lag_10"] = df["silver_close"].shift(10)
    df["lag_20"] = df["silver_close"].shift(20)

    df["ret_1"]  = df["silver_close"].pct_change(1)
    df["ret_5"]  = df["silver_close"].pct_change(5)
    df["ret_20"] = df["silver_close"].pct_change(20)

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()
    if df.empty:
        return None, None

    X_last = df[
        ["silver_close", "usd_inr", "lag_1", "lag_5", "lag_10", "lag_20",
         "ret_1", "ret_5", "ret_20", "dayofweek", "month"]
    ].iloc[[-1]]

    latest = df.iloc[-1]
    return X_last, latest

def build_price_cards(base_inr_kg, premium_per_kg, purity_factor):
    """
    Returns predicted prices for 1g/10g/100g.
    We convert:
      base INR/kg -> INR/g
      premium INR/kg -> INR/g
      apply purity factor
    """
    base_per_g = (base_inr_kg / 1000.0) * purity_factor
    premium_per_g = (premium_per_kg / 1000.0)

    def price_for(g):
        return round((base_per_g + premium_per_g) * g, 2)

    return {
        "per_g": round(base_per_g + premium_per_g, 2),
        "p1": price_for(1),
        "p10": price_for(10),
        "p100": price_for(100),
    }

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    # Price chart (last 48 points)
    series = fetch_hourly_series(days="5d")
    labels, values = [], []
    if series is not None and not series.empty:
        s = series.tail(48)
        labels = [str(x)[11:16] for x in s.index]  # HH:MM
        values = [round(float(v), 3) for v in s.values]

    if request.method == "POST":
        state = request.form.get("state")
        horizon = request.form.get("horizon")  # 1h / 1d / 1m
        purity = request.form.get("purity")    # 999/925/900/800

        premium = STATE_PREMIUM.get(state, 0)
        purity_factor = SILVER_PURITY.get(purity, 1.0)

        try:
            if horizon in ["1h", "1d"]:
                X_last, latest = fetch_hourly_features()
                if X_last is None:
                    flash("Could not fetch hourly market data right now. Please try again.", "error")
                    return redirect(url_for("index"))

                if horizon == "1h":
                    direction, confidence, proba_up = predict_with_model(model_1h, X_last)
                    horizon_title = "Next Hour"
                else:
                    direction, confidence, proba_up = predict_with_model(model_1d, X_last)
                    horizon_title = "Next Day (24h)"

                base_inr_kg = usd_oz_to_inr_kg(latest["silver_close"], latest["usd_inr"])

            else:
                X_last, latest = fetch_daily_features_for_month()
                if X_last is None:
                    flash("Could not fetch daily market data right now. Please try again.", "error")
                    return redirect(url_for("index"))

                direction, confidence, proba_up = predict_with_model(model_1m, X_last)
                horizon_title = "Next Month (approx)"
                base_inr_kg = usd_oz_to_inr_kg(latest["silver_close"], latest["usd_inr"])

            price_cards = build_price_cards(base_inr_kg, premium, purity_factor)

            result = {
                "state": state,
                "horizon": horizon_title,
                "direction": direction,
                "confidence": confidence,
                "proba_up": proba_up,
                "purity": purity,
                "premium_per_kg": premium,
                "base_inr_kg": round(base_inr_kg, 2),
                "prices": price_cards
            }

        except Exception:
            flash("Something went wrong while calculating prediction. Please try again.", "error")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        states=sorted(STATE_PREMIUM.keys()),
        result=result,
        labels=labels,
        values=values,
        purity_map=SILVER_PURITY,
        social=SOCIAL
    )

@app.route("/about")
def about():
    return render_template("about.html", social=SOCIAL)

@app.route("/reviews", methods=["GET", "POST"])
def reviews():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        rating = int(request.form.get("rating", "5"))
        message = request.form.get("message", "").strip()

        if not name or not message:
            flash("Please enter your name and review message.", "error")
            return redirect(url_for("reviews"))

        if rating < 1 or rating > 5:
            flash("Rating must be between 1 and 5.", "error")
            return redirect(url_for("reviews"))

        add_review(name, rating, message)
        flash("Thanks! Your review has been added.", "success")
        return redirect(url_for("reviews"))

    all_reviews = get_reviews(limit=60)
    return render_template("reviews.html", reviews=all_reviews, social=SOCIAL)

if __name__ == "__main__":
    app.run(debug=True)
