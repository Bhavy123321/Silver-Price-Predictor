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
# SOCIAL LINKS
# -------------------------
SOCIAL = {
    "github": "https://github.com/YOUR_GITHUB_USERNAME",
    "linkedin": "https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/"
}

# -------------------------
# INDIA STATE PREMIUM (₹/kg)
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
    "Ladakh": 720,
    "Lakshadweep": 650,
    "Madhya Pradesh": 720,
    "Maharashtra": 1000,
    "Odisha": 700,
    "Punjab": 850,
    "Rajasthan": 750,
    "Tamil Nadu": 600,
    "Telangana": 650,
    "Uttar Pradesh": 850,
    "Uttarakhand": 800,
    "West Bengal": 650,
}

# -------------------------
# SILVER PURITY
# -------------------------
SILVER_PURITY = {
    "999": 1.0,
    "925": 0.925,
    "900": 0.9,
    "800": 0.8,
}

# -------------------------
# LOAD MODELS
# -------------------------
model_1h = joblib.load("models/model_next_hour.joblib")
model_1d = joblib.load("models/model_next_day.joblib")
model_1m = joblib.load("models/model_next_month.joblib")

# -------------------------
# UTILS
# -------------------------
def usd_oz_to_inr_kg(price_usd_oz, usd_inr):
    return float(price_usd_oz) * float(usd_inr) * (1000 / 31.1034768)

def safe_close(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].iloc[:, 0].dropna()
    return df["Close"].dropna()

def ensure_features(model, X):
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in X.columns:
                X[c] = 0
        return X[cols]
    return X

def predict(model, X):
    X = ensure_features(model, X)
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0][1]
    return ("UP" if pred == 1 else "DOWN", round(prob * 100, 2))

# -------------------------
# FETCH FEATURES
# -------------------------
def fetch_daily():
    s = yf.download("SI=F", period="180d", interval="1d", auto_adjust=True, progress=False)
    u = yf.download("USDINR=X", period="180d", interval="1d", auto_adjust=True, progress=False)

    sc = safe_close(s)
    uc = safe_close(u)
    if sc is None or uc is None:
        return None, None

    df = pd.concat([sc.rename("silver"), uc.rename("usd")], axis=1).dropna()
    df["lag1"] = df["silver"].shift(1)
    df["lag2"] = df["silver"].shift(2)
    df["ret1"] = df["silver"].pct_change(1)
    df["dow"] = df.index.dayofweek
    df = df.dropna()

    return df.iloc[[-1]][["silver", "usd", "lag1", "lag2", "ret1", "dow"]], df.iloc[-1]

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        state = request.form["state"]
        purity = request.form["purity"]
        horizon = request.form["horizon"]

        premium = STATE_PREMIUM[state]
        purity_factor = SILVER_PURITY[purity]

        X, latest = fetch_daily()
        if X is None:
            flash("Market data unavailable", "error")
            return redirect("/")

        direction, confidence = predict(model_1d, X)

        base_inr_kg = usd_oz_to_inr_kg(latest["silver"], latest["usd"])
        current_g = round((base_inr_kg / 1000) * purity_factor, 2)

        final_g = round(current_g + premium / 1000, 2)

        result = {
            "state": state,
            "direction": direction,
            "confidence": confidence,
            "current_g": current_g,
            "base_kg": round(base_inr_kg, 2),
            "premium": premium,
            "final_g": final_g,
            "p1": final_g,
            "p10": round(final_g * 10, 2),
            "p100": round(final_g * 100, 2),
        }

    return render_template(
        "index.html",
        states=sorted(STATE_PREMIUM.keys()),
        result=result,
        social=SOCIAL,
    )

if __name__ == "__main__":
    app.run(debug=True)
