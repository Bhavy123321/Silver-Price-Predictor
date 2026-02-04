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
    "github": "https://github.com/YOUR_GITHUB_USERNAME",
    "linkedin": "https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/"
}

# -------------------------
# ALL INDIA: States + UTs
# Premium is demo premium (₹ per kg)
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

    # UTs
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
# INDIA RETAIL ESTIMATE (simple model)
# These are just to approximate Google-like retail range
# -------------------------
DEFAULT_GST_RATE = 0.03           # 3% GST (approx)
DEFAULT_RETAIL_MARGIN_KG = 25000  # extra ₹/kg margin seen in retail markets (adjust as you want)

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
    # 1 kg = 32.1507466 troy ounces
    oz_per_kg = 1000.0 / 31.1034768
    return float(price_usd_per_oz) * float(usd_inr) * oz_per_kg

def safe_close_series(df):
    """
    yfinance sometimes returns MultiIndex columns.
    We always prefer Close (NOT adjusted) because futures can behave weird with auto_adjust.
    """
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        # If MultiIndex, Close is a DataFrame; take first column
        if "Close" in df:
            return df["Close"].iloc[:, 0].dropna()
        return None
    if "Close" in df.columns:
        return df["Close"].dropna()
    return None

def prepare_X_for_model(model, X_df: pd.DataFrame):
    """
    Fixes: "X does not have valid feature names" and feature mismatch errors.
    - If model has feature_names_in_, reindex to that exact order.
    - Fill missing with 0.
    """
    if X_df is None or X_df.empty:
        return None

    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            X_df = X_df.reindex(columns=cols, fill_value=0)
            return X_df
        return X_df
    except Exception:
        # fallback: raw numpy
        return X_df.values

def predict_with_model(model, X_row: pd.DataFrame):
    """
    Returns: direction, confidence%, proba_up%
    direction: UP or DOWN
    confidence: model confidence for predicted class
    proba_up: probability of UP
    """
    X_ready = prepare_X_for_model(model, X_row)
    if X_ready is None:
        raise ValueError("No features to predict.")

    pred = int(model.predict(X_ready)[0])

    # proba_up
    if hasattr(model, "predict_proba"):
        proba_up = float(model.predict_proba(X_ready)[0][1])
    else:
        # Some models don’t support proba
        proba_up = 0.5

    direction = "UP" if pred == 1 else "DOWN"
    confidence = proba_up if pred == 1 else (1 - proba_up)

    return direction, round(confidence * 100, 2), round(proba_up * 100, 2)

def fetch_hourly_series(days="5d"):
    """
    For chart
    """
    try:
        silver = yf.download("SI=F", period=days, interval="1h", auto_adjust=False, progress=False)
        series = safe_close_series(silver)
        return series
    except Exception:
        return None

def fetch_hourly_features():
    """
    Features for Next Hour model
    """
    silver = yf.download("SI=F", period="5d", interval="1h", auto_adjust=False, progress=False)
    usdinr = yf.download("USDINR=X", period="5d", interval="1h", auto_adjust=False, progress=False)

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

def fetch_daily_features_for_day():
    """
    FIX: Next Day uses DAILY data (stable)
    """
    silver = yf.download("SI=F", period="200d", interval="1d", auto_adjust=False, progress=False)
    usdinr = yf.download("USDINR=X", period="200d", interval="1d", auto_adjust=False, progress=False)

    silver_close = safe_close_series(silver)
    usd_inr = safe_close_series(usdinr)

    if silver_close is None or usd_inr is None:
        return None, None

    df = pd.concat([silver_close.rename("silver_close"), usd_inr.rename("usd_inr")], axis=1).dropna()

    df["lag_1"] = df["silver_close"].shift(1)
    df["lag_2"] = df["silver_close"].shift(2)
    df["lag_3"] = df["silver_close"].shift(3)

    df["ret_1"] = df["silver_close"].pct_change(1)
    df["ret_5"] = df["silver_close"].pct_change(5)

    df["dayofweek"] = df.index.dayofweek

    df = df.dropna()
    if df.empty:
        return None, None

    X_last = df[["silver_close", "usd_inr", "lag_1", "lag_2", "lag_3", "ret_1", "ret_5", "dayofweek"]].iloc[[-1]]
    latest = df.iloc[-1]
    return X_last, latest

def fetch_daily_features_for_month():
    """
    Next Month features from daily data (2 years)
    """
    silver = yf.download("SI=F", period="2y", interval="1d", auto_adjust=False, progress=False)
    usdinr = yf.download("USDINR=X", period="2y", interval="1d", auto_adjust=False, progress=False)

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

def build_predicted_prices(base_inr_kg, premium_per_kg, purity_factor):
    """
    Predicted (Estimated) prices for 1g / 10g / 100g, based on:
    base market converted + state premium + purity factor
    """
    base_per_g = (base_inr_kg / 1000.0) * purity_factor
    premium_per_g = (premium_per_kg / 1000.0)

    per_g = base_per_g + premium_per_g

    def price_for(g):
        return round(per_g * g, 2)

    return {
        "per_g": round(per_g, 2),
        "p1": price_for(1),
        "p10": price_for(10),
        "p100": price_for(100),
    }

def build_india_retail_estimate(global_inr_kg, state_premium_kg, gst=DEFAULT_GST_RATE, retail_margin_kg=DEFAULT_RETAIL_MARGIN_KG):
    """
    India retail estimate (rough):
    global reference + state premium + retail margin, then GST
    """
    pre_tax = float(global_inr_kg) + float(state_premium_kg) + float(retail_margin_kg)
    post_tax = pre_tax * (1 + float(gst))
    return round(post_tax, 2)

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    # Trend chart values
    series = fetch_hourly_series(days="5d")
    labels, values = [], []
    if series is not None and not series.empty:
        s = series.tail(48)
        labels = [str(x)[11:16] for x in s.index]  # HH:MM
        values = [round(float(v), 3) for v in s.values]

    if request.method == "POST":
        state = request.form.get("state")
        horizon = request.form.get("horizon")   # 1h / 1d / 1m
        purity = request.form.get("purity")     # 999/925/900/800

        premium = STATE_PREMIUM.get(state, 0)
        purity_factor = SILVER_PURITY.get(purity, 1.0)

        try:
            # NEXT HOUR
            if horizon == "1h":
                X_last, latest = fetch_hourly_features()
                if X_last is None:
                    flash("Could not fetch hourly market data right now. Please try again.", "error")
                    return redirect(url_for("index"))

                direction, confidence, proba_up = predict_with_model(model_1h, X_last)
                horizon_title = "Next Hour"
                global_inr_kg = usd_oz_to_inr_kg(latest["silver_close"], latest["usd_inr"])

            # NEXT DAY (FIXED)
            elif horizon == "1d":
                X_last, latest = fetch_daily_features_for_day()
                if X_last is None:
                    flash("Could not fetch daily market data right now. Please try again.", "error")
                    return redirect(url_for("index"))

                direction, confidence, proba_up = predict_with_model(model_1d, X_last)
                horizon_title = "Next Day"
                global_inr_kg = usd_oz_to_inr_kg(latest["silver_close"], latest["usd_inr"])

            # NEXT MONTH
            else:
                X_last, latest = fetch_daily_features_for_month()
                if X_last is None:
                    flash("Could not fetch daily market data right now. Please try again.", "error")
                    return redirect(url_for("index"))

                direction, confidence, proba_up = predict_with_model(model_1m, X_last)
                horizon_title = "Next Month (approx)"
                global_inr_kg = usd_oz_to_inr_kg(latest["silver_close"], latest["usd_inr"])

            # Predicted (Estimated) — per gram etc.
            predicted_prices = build_predicted_prices(global_inr_kg, premium, purity_factor)

            # India retail estimate (to match Google-like higher range)
            india_retail_kg = build_india_retail_estimate(global_inr_kg, premium)

            result = {
                "state": state,
                "horizon": horizon_title,
                "direction": direction,
                "confidence": confidence,
                "proba_up": proba_up,
                "purity": purity,

                # global reference (converted)
                "global_inr_kg": round(float(global_inr_kg), 2),

                # predicted estimates (with purity + premium)
                "predicted": predicted_prices,

                # state premium
                "premium_per_kg": premium,

                # india retail estimate (rough)
                "india_retail_kg": india_retail_kg,
                "gst_rate": DEFAULT_GST_RATE,
                "retail_margin_kg": DEFAULT_RETAIL_MARGIN_KG
            }

        except Exception as e:
            print("PREDICTION ERROR:", str(e))  # helpful on Railway logs
            flash("Something went wrong while calculating prediction. Please try again.", "error")
            return redirect(url_for("index"))

    return render_template(
        "index.html",
        states=sorted(STATE_PREMIUM.keys()),
        result=result,
        labels=labels,
        values=values,
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

# ✅ For Railway/Gunicorn: it needs a top-level "app"
# gunicorn will use: app:app
if __name__ == "__main__":
    app.run(debug=True)
