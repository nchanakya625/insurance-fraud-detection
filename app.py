"""
=============================================================================
  INSURANCE FRAUD DETECTION — Flask Web Application
=============================================================================
Loads the trained model and provides a web UI for fraud prediction.
"""

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Load saved artifacts ───────────────────────────────────────────────────────
model           = joblib.load("best_model.pkl")
scaler          = joblib.load("scaler.pkl")
label_encoders  = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ── Feature metadata for the form ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "months_as_customer", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "bodily_injuries", "witnesses", "total_claim_amount", "auto_year",
]

CATEGORICAL_FEATURES = {
    "policy_state":          ["OH", "IN", "IL"],
    "policy_csl":            ["100/300", "250/500", "500/1000"],
    "insured_sex":           ["MALE", "FEMALE"],
    "insured_education_level": ["JD", "High School", "College", "Masters",
                                 "Associate", "MD", "PhD"],
    "insured_occupation":    ["craft-repair", "machine-op-inspct", "sales",
                               "armed-forces", "tech-support", "prof-specialty",
                               "other-service", "exec-managerial",
                               "priv-house-serv", "transport-moving",
                               "handlers-cleaners", "protective-serv",
                               "farming-fishing", "adm-clerical"],
    "insured_hobbies":       ["sleeping", "reading", "board-games",
                               "bungie-jumping", "base-jumping", "golf",
                               "camping", "dancing", "skydiving", "cross-fit",
                               "chess", "exercise", "hiking", "yachting",
                               "paintball", "movies", "kayaking", "polo",
                               "basketball", "video-games"],
    "insured_relationship":  ["husband", "other-relative", "own-child",
                               "unmarried", "wife", "not-in-family"],
    "incident_type":         ["Single Vehicle Collision", "Vehicle Theft",
                               "Multi-vehicle Collision", "Parked Car"],
    "collision_type":        ["Side Collision", "Rear Collision",
                               "Front Collision", "?"],
    "incident_severity":     ["Minor Damage", "Major Damage", "Total Loss",
                               "Trivial Damage"],
    "authorities_contacted": ["Police", "Fire", "Other", "Ambulance", "None"],
    "incident_state":        ["SC", "WV", "VA", "NY", "PA", "NC", "OH"],
    "property_damage":       ["YES", "NO", "?"],
    "police_report_available": ["YES", "NO", "?"],
    "auto_make":             ["Saab", "Mercedes", "Dodge", "Chevrolet",
                               "Accura", "Nissan", "Toyota", "Ford", "BMW",
                               "Audi", "Suburu", "Jeep", "Honda", "Volkswagen"],
}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the prediction form."""
    return render_template("index.html",
                           numeric_features=NUMERIC_FEATURES,
                           categorical_features=CATEGORICAL_FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    """Retrieve form values, run prediction, show result."""
    try:
        input_data = {}

        # Collect numeric inputs
        for feat in NUMERIC_FEATURES:
            val = request.form.get(feat, "0")
            input_data[feat] = float(val) if val else 0.0

        # Collect categorical inputs & encode
        for feat, options in CATEGORICAL_FEATURES.items():
            val = request.form.get(feat, options[0])
            if feat in label_encoders:
                le = label_encoders[feat]
                if val in le.classes_:
                    input_data[feat] = le.transform([val])[0]
                else:
                    input_data[feat] = 0
            else:
                input_data[feat] = 0

        # Build DataFrame in correct column order
        row = {}
        for col in feature_columns:
            row[col] = input_data.get(col, 0)

        df = pd.DataFrame([row])
        df_scaled = scaler.transform(df)

        # Get probability of fraud (class 1)
        prob_fraud = model.predict_proba(df_scaled)[0][1]
        
        # Use a lower threshold (0.3) since data is imbalanced and it improves recall
        THRESHOLD = 0.3
        is_fraud = prob_fraud >= THRESHOLD

        prediction = "🚨 FRAUDULENT" if is_fraud else "✅ NOT FRAUDULENT"

        return render_template("submit.html", prediction=prediction, is_fraud=is_fraud)

    except Exception as e:
        return render_template("submit.html",
                               prediction=f"Error: {str(e)}", is_fraud=False)


if __name__ == "__main__":
    print("\n🌐 Starting Insurance Fraud Detection Web App...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
