import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/credit_risk_pipeline.joblib")

model = load_model()

# ---------------- UI ----------------
st.title("💳 Credit Risk Prediction")

# Numerical inputs
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income", 0.0, 10_000_000.0, 500000.0)
employment_length = st.number_input("Employment Length (years)", 0, 50, 5)
loan_amount = st.number_input("Loan Amount", 0.0, 5_000_000.0, 200000.0)
interest_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
credit_history_length = st.number_input("Credit History Length (years)", 0, 50, 8)

# Categorical → one-hot manually
home_ownership = st.selectbox("Home Ownership", ["own", "rent"])
loan_purpose = st.selectbox(
    "Loan Purpose",
    ["education", "home", "medical", "personal"]
)

# ---------------- PREDICTION ----------------
if st.button("Predict Default Risk"):
    X = pd.DataFrame([{
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "credit_history_length": credit_history_length,

        # one-hot encoding (HOME OWNERSHIP)
        "home_ownership_own": 1 if home_ownership == "own" else 0,
        "home_ownership_rent": 1 if home_ownership == "rent" else 0,

        # one-hot encoding (LOAN PURPOSE)
        "loan_purpose_education": 1 if loan_purpose == "education" else 0,
        "loan_purpose_home": 1 if loan_purpose == "home" else 0,
        "loan_purpose_medical": 1 if loan_purpose == "medical" else 0,
        "loan_purpose_personal": 1 if loan_purpose == "personal" else 0,
    }])

    prob = model.predict_proba(X)[0][1]

    st.subheader("Result")
    st.metric("Default Risk Probability", f"{prob:.2%}")

    if prob >= 0.7:
        st.error("🚨 High Risk — Likely Default")
    elif prob >= 0.4:
        st.warning("⚠️ Medium Risk — Manual Review Needed")
    else:
        st.success("✅ Low Risk — Acceptable")
